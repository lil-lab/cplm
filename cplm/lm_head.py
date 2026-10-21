"""Vocab-tiled LM-head reductions.

The CPLM loss needs only three numbers per position from the LM head: the
log-partition over the vocabulary, the logit of the gold token and the logit of
the <copy> token. We project the hidden states onto the vocabulary one tile of
``tile`` rows at a time and reduce on the fly, so the [B, T, V] logits tensor is
never materialized. Under torch.compile, inductor recomputes each tile in the
backward pass, so backward memory also stays at one tile.
"""

import torch
import torch.nn.functional as F


def _reductions(hidden, weight, labels, copy_id, tile):
    H = hidden.shape[-1]
    x = hidden.reshape(-1, H)                                   # [N, H]
    y = labels.reshape(-1)                                      # [N]
    V = weight.shape[0]

    run_max = x.new_full((x.shape[0],), float("-inf"), dtype=torch.float32)
    run_sum = x.new_zeros((x.shape[0],), dtype=torch.float32)
    logit_y = x.new_zeros((x.shape[0],), dtype=torch.float32)
    logit_copy = None

    for v0 in range(0, V, tile):
        v1 = min(v0 + tile, V)
        z = F.linear(x, weight[v0:v1]).float()                  # [N, v1 - v0]

        # streaming logsumexp
        new_max = torch.maximum(run_max, z.amax(dim=-1))
        run_sum = run_sum * (run_max - new_max).exp() + (z - new_max[:, None]).exp().sum(-1)
        run_max = new_max

        in_tile = (y >= v0) & (y < v1)
        picked = z.gather(1, (y - v0).clamp(0, v1 - v0 - 1)[:, None]).squeeze(1)
        logit_y = torch.where(in_tile, picked, logit_y)

        if v0 <= copy_id < v1:                                  # static: copy_id is an int
            logit_copy = z[:, copy_id - v0]

    lse = run_max + run_sum.log()
    shape = labels.shape
    return lse.view(shape), logit_y.view(shape), logit_copy.view(shape)


_reductions_compiled = torch.compile(_reductions, dynamic=False)


def lm_head_reductions(hidden, weight, labels, copy_id, tile=8192, compile=True):
    """Returns (logsumexp, logit at label, logit at <copy>), each shaped like ``labels``."""
    fn = _reductions_compiled if compile else _reductions
    return fn(hidden, weight, labels, copy_id, tile)
