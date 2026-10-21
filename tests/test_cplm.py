"""CPU checks of the pieces that carry the method: document bookkeeping, the
vocab-tiled LM head, and the copy-mixture loss against a direct computation."""

import torch

from cplm.data import prepare_batch
from cplm.lm_head import lm_head_reductions
from cplm.model import CPLM, EPS, EPS_ALPHA

EOS = 0


def test_prepare_batch():
    batch = torch.tensor([[5, 6, EOS, 7, 8, EOS, 9, 4]])
    x, y, pos, doc = prepare_batch(batch, EOS)
    assert x.tolist() == [[5, 6, EOS, 7, 8, EOS, 9]]
    assert y.tolist() == [[6, EOS, 7, 8, EOS, 9, 4]]
    assert pos.tolist() == [[0, 1, 2, 0, 1, 2, 0]]
    assert doc.tolist() == [[0, 0, 0, 1, 1, 1, 2]]


def test_lm_head_reductions():
    torch.manual_seed(0)
    h, W = torch.randn(2, 5, 8), torch.randn(37, 8)
    y = torch.randint(0, 36, (2, 5))
    lse, ly, lc = lm_head_reductions(h, W, y, copy_id=36, tile=10, compile=False)
    logits = h @ W.T
    assert torch.allclose(lse, logits.logsumexp(-1), atol=1e-5)
    assert torch.allclose(ly, logits.gather(-1, y[..., None])[..., 0], atol=1e-5)
    assert torch.allclose(lc, logits[..., 36], atol=1e-5)


class _Backbone(torch.nn.Module):
    """Stands in for the HF decoder: an embedding + tied head, no attention."""

    def __init__(self, V, H):
        super().__init__()
        self.config = type("C", (), {"hidden_size": H})()
        self.emb = torch.nn.Embedding(V, H)
        outer = self

        class Inner(torch.nn.Module):
            def forward(self, ids, position_ids=None, **kw):
                return type("O", (), {"last_hidden_state": outer.emb(ids)})()
        self.model = Inner()

    def get_output_embeddings(self):
        return self.emb


def test_copy_mixture_loss():
    torch.manual_seed(0)
    V, H = 12, 16
    model = CPLM(_Backbone(V + 1, H), copy_id=V, compile_loss=False)
    batch = torch.tensor([[3, 4, 5, 3, EOS, 4, 3, 4, 5],
                          [6, 7, 6, 7, 6, EOS, 7, 6, 7]])
    x, y, pos, doc = prepare_batch(batch, EOS)
    loss = model(x, y, pos, doc)["loss"]

    # Direct computation, position by position.
    h = model.backbone.emb(x)
    p_lm_all = (h @ model.backbone.emb.weight.T).softmax(-1)
    q, k = model.q_head(h), model.k_head(h)
    nll = []
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            src = [j for j in range(t) if doc[b, j] == doc[b, t]]
            s = torch.stack([q[b, t] @ model.k_sink] + [q[b, t] @ k[b, j] for j in src]) * H ** -0.5
            a = s.softmax(0)
            alpha, p_lm = p_lm_all[b, t, V], p_lm_all[b, t, y[b, t]]
            p_copy = sum(a[1 + i] for i, j in enumerate(src) if x[b, j] == y[b, t])
            p = p_lm * (1 + alpha * a[0] / (1 - alpha).clamp_min(EPS_ALPHA)) + alpha * p_copy
            nll.append(-(p + EPS).log())
    assert torch.allclose(loss, torch.stack(nll).mean(), atol=1e-5)
