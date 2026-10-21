"""CPLM: a causal LM that can copy tokens from the current document.

The backbone is a SmolLM3-layout decoder (RoPE, optionally tied embeddings,
optional sliding-window attention). On top of its final hidden state h_t sit:

  * the LM head, whose vocabulary has one extra token <copy> (id = vocab_size);
  * a single-head copy pointer, q_t = W_q h_t, k_j = W_k h_j, attending over the
    strictly-earlier positions j < t of the SAME document plus a learned sink k_s.

With a_j the pointer weights and a_s the sink weight (a_s + sum_j a_j = 1) and
alpha = p_lm(<copy>), the probability of the next token y is

    p(y) = p_lm(y) * (1 + alpha * a_s / (1 - alpha)) + alpha * sum_{j: x_j = y} a_j

i.e. the <copy> mass goes to the pointed-at tokens, and whatever the pointer
puts on the sink is handed back to the LM proportionally ("rollback"). This is
a proper distribution over the real vocabulary, so p(<copy>) itself is never a
target.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, SmolLM3Config

from .lm_head import lm_head_reductions

EPS_ALPHA = 1e-6   # floor on the non-copy LM mass (1 - alpha)
EPS = 1e-9         # floor inside the log


def build_backbone(cfg):
    """SmolLM3 decoder from the model section of the config, randomly initialized.

    ``sliding_window`` null -> full attention everywhere. With a window,
    ``full_attn_every`` = k > 0 makes every k-th layer and the last layer full
    attention (hybrid); 0 keeps every layer windowed. ``nope_every`` = k > 0
    drops RoPE from every k-th layer (SmolLM3's NoPE layers); 0 keeps RoPE
    everywhere.
    """
    L = cfg["num_layers"]
    nope = cfg.get("nope_every", 0)
    window = cfg.get("sliding_window")
    every = cfg.get("full_attn_every", 0)
    if window is None:
        layer_types = ["full_attention"] * L
    else:
        layer_types = ["full_attention" if every and ((i + 1) % every == 0 or i == L - 1)
                       else "sliding_attention" for i in range(L)]
    config = SmolLM3Config(
        vocab_size=cfg["vocab_size"] + 1,                       # + <copy>
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=L,
        num_attention_heads=cfg["num_heads"],
        num_key_value_heads=cfg["num_kv_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        initializer_range=cfg["init_std"],
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": float(cfg.get("rope_theta", 10000.0))},
        no_rope_layers=[int(not nope or (i + 1) % nope != 0) for i in range(L)],   # 1 = RoPE
        use_sliding_window=window is not None,
        sliding_window=window,
        layer_types=layer_types,
        tie_word_embeddings=cfg.get("tie_embeddings", True),
        bos_token_id=cfg["eos_id"],
        eos_token_id=cfg["eos_id"],
        pad_token_id=None,
        use_cache=False,
    )
    return AutoModelForCausalLM.from_config(config, attn_implementation=cfg["attn_implementation"])


def varlen_kwargs(position_ids):
    """Flash-attention varlen arguments for a (1, N) row of packed documents.

    Every position with id 0 starts a segment, so attention becomes block-diagonal
    over documents (and over the original batch rows, which all start at 0).
    """
    flat = position_ids.reshape(-1)
    starts = (flat == 0).nonzero().view(-1).to(torch.int32)
    cu = torch.cat([starts, starts.new_tensor([flat.numel()])])
    max_len = cu.diff().max()
    return {"cu_seq_lens_q": cu, "cu_seq_lens_k": cu,
            "max_length_q": max_len, "max_length_k": max_len}


class CPLM(nn.Module):
    def __init__(self, backbone, copy_id, compile_loss=True):
        super().__init__()
        self.backbone = backbone
        self.copy_id = int(copy_id)
        self.compile_loss = compile_loss
        H = backbone.config.hidden_size
        self.q_head = nn.Linear(H, H, bias=False)
        self.k_head = nn.Linear(H, H, bias=False)
        self.k_sink = nn.Parameter(torch.randn(H) * H ** -0.5)
        self.scale = H ** -0.5

    def forward(self, input_ids, labels, position_ids, doc_ids):
        """Mean next-token NLL under the copy mixture, plus detached diagnostics.

        input_ids, labels, position_ids (reset to 0 after every EOS), doc_ids: [B, T].
        """
        B, T = input_ids.shape
        # Backbone on the flattened (1, B*T) row so documents are varlen segments:
        # transformers only infers packing from position_ids at batch size 1.
        pos = position_ids.reshape(1, B * T)
        h = self.backbone.model(input_ids.reshape(1, B * T), position_ids=pos,
                                **varlen_kwargs(pos)).last_hidden_state.view(B, T, -1)

        # Copy pointer over strictly-earlier positions of the same document + sink.
        q, k = self.q_head(h), self.k_head(h)
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale                  # [B, T, T]
        past = torch.ones(T, T, dtype=torch.bool, device=h.device).tril(-1)
        allowed = past & (doc_ids[:, :, None] == doc_ids[:, None, :])
        scores = scores.masked_fill(~allowed, -1e9)
        sink = (q @ self.k_sink) * self.scale                                      # [B, T]
        probs = torch.cat([sink[..., None], scores], dim=-1).softmax(dim=-1)
        a_sink, attn = probs[..., 0], probs[..., 1:]
        p_copy = (attn * (input_ids[:, None, :] == labels[:, :, None])).sum(-1)   # [B, T]

        # Mixture with the LM distribution.
        head_w = self.backbone.get_output_embeddings().weight
        lse, logit_y, logit_copy = lm_head_reductions(h, head_w, labels, self.copy_id,
                                                      compile=self.compile_loss)
        p_lm = (logit_y - lse).exp()
        alpha = (logit_copy - lse).exp()
        rollback = alpha * a_sink / (1.0 - alpha).clamp_min(EPS_ALPHA)
        p = p_lm * (1.0 + rollback) + alpha * p_copy
        loss = -(p + EPS).log().mean()
        return {"loss": loss,
                "copy_rate": alpha.detach().mean(),
                "sink_rate": a_sink.detach().mean()}
