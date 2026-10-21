# CPLM: Copy-Paste Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2610.19247-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.19247)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffd21e)](https://huggingface.co/collections/nthngdy/cplm-6acd1a63a8641f3c2b8488f3)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Minimal code to pretrain **Copy-Paste Language Models** from scratch, from the paper
*Copy-Paste Language Models* (Nathan Godey & Yoav Artzi, Cornell University).

A CPLM adds a lightweight **Copy-Paste (CP) head** next to the LM head. The CP head
points at an earlier token of the current document and copies it. It is triggered by
an extra `<copy>` token in the LM vocabulary and can fall back to the LM head through
a learned sink. For the next token `y`:

```
p(y) = p_lm(y) * (1 + alpha * a_sink / (1 - alpha)) + alpha * sum_{j < t, same doc, x_j = y} a_j
```

where `alpha = p_lm(<copy>)`, and `a_j`, `a_sink` are the CP head's attention weights
over earlier positions and over the sink (`a_sink + sum_j a_j = 1`). Mass the CP head
puts on the sink goes back to the LM, so `p` is a distribution over the real
vocabulary. The model is trained end to end with the next-token loss on `p` only.
See [`cplm/model.py`](cplm/model.py).

## Repository

```
cplm/model.py      backbone builder, CP head, mixture loss
cplm/lm_head.py    vocabulary-tiled LM-head reductions (no [B, T, V] logits)
cplm/data.py       litdata loaders, per-document positions and ids
cplm/train.py      Lightning Fabric training loop (FSDP/DDP, torch.compile, resume)
configs/           135M (20B tokens) and 1.7B (200B tokens) sliding-window CPLMs
scripts/           data preparation, Clariden environment setup
slurm/             SLURM launcher with automatic requeue
tests/             CPU tests
```

## Setup

Requires PyTorch >= 2.9 and FlashAttention (3 on Hopper): documents in a packed
sequence are kept apart with varlen attention.

```bash
pip install -e ".[data,test]"
```

then install FlashAttention-3 for your torch/CUDA version (built from the `hopper/`
directory of [flash-attention](https://github.com/Dao-AILab/flash-attention), or a
prebuilt wheel).

`scripts/setup_clariden.sh` is the recipe we use on GH200 (aarch64) nodes.

## Data

Training reads a [litdata](https://github.com/Lightning-AI/litdata) dataset of packed
uint16 token ids, each document followed by EOS. To build it from FineWeb-Edu with the
SmolLM2 tokenizer:

```bash
python scripts/prepare_data.py --out data/fineweb-edu-350BT --subset sample/350BT --workers 32
```

`train.data_path` may also point to a parent directory of several such datasets,
which are then combined. 0.5% of the data is held out (fixed seed) for validation.

## Training

```bash
# one node, all its GPUs
python -m cplm.train configs/cplm_135M_sw256.json

# SLURM: one task per GPU
sbatch -A <account> slurm/train.sbatch configs/cplm_135M_sw256.json
sbatch -A <account> --nodes=4 slurm/train.sbatch configs/cplm_1p7B_sw256.json
```

Extra srun options (a container, a uenv) can be passed in `$SRUN_ARGS`. On SLURM,
the job stops 5 minutes before its time limit, requeues itself, and resumes from
the newest complete checkpoint, so up to `last_ckpt_every` steps are recomputed.
A run started again with the same run id (`$RUN_ID`, else `$SLURM_JOB_ID`)
resumes the same way and appends to the same wandb run. Resuming needs the same
world size and `num_workers`.

### What is fixed

- one CP head; query/key dimension = hidden size; learned sink
- the CP head only attends to strictly earlier tokens of the **same document**
- documents are isolated in the backbone too (positions restart after every EOS)
- SmolLM3-layout decoder, `<copy>` id = `vocab_size`
- fp32 weights, bf16-mixed compute, AdamW on all parameters, warmup-stable-decay
  schedule (linear warmup, constant, cosine cooldown)

### Configuration

| Section | Key | Meaning |
|---|---|---|
| `model` | `hidden_size`, `intermediate_size`, `num_layers`, `num_heads`, `num_kv_heads`, `init_std` | backbone size and init |
| | `sliding_window` | window size; `null` = full attention |
| | `full_attn_every` | with a window, every k-th and the last layer use full attention (0 = none) |
| | `tie_embeddings` | tie input embeddings and LM head |
| | `rope_theta`, `nope_every` | RoPE base; drop RoPE from every k-th layer (0 = RoPE everywhere) |
| | `vocab_size`, `eos_id`, `max_position_embeddings`, `attn_implementation` | tokenizer and attention backend |
| `optim` | `lr`, `betas`, `eps`, `weight_decay`, `grad_clip` | AdamW |
| | `warmup_steps`, `cooldown_steps`, `min_lr_ratio` | WSD schedule; final LR = `min_lr_ratio * lr` |
| `train` | `seq_length`, `batch_size`, `micro_batch_size`, `steps` | `batch_size` counts sequences, globally; gradient accumulation is derived |
| | `strategy`, `reduce_dtype`, `compile` | `fsdp` (shard grads and optimizer state), `fsdp_full`, or `ddp`; FSDP gradient-reduction dtype |
| | `data_path`, `num_workers`, `seed` | data |
| | `ckpt_root`, `ckpt_every`, `last_ckpt_every`, `val_every`, `val_batches`, `log_every`, `wandb_project` | checkpoints and logging (`wandb_project: null` disables wandb) |

The two configs reproduce the released sliding-window (`sw`) CPLMs: 135M on 20B
tokens (19k steps of 512 x 2,048 tokens) and 1.7B on 200B tokens (95k steps of
512 x 4,096 tokens). Set `sliding_window: null` for the `full` backbones and
`full_attn_every: 4` for the `hybrid` ones.

## Tests

```bash
python -m pytest tests
```

## Citation

```bibtex
@misc{godey2026copypastelanguagemodels,
      title={Copy-Paste Language Models},
      author={Nathan Godey and Yoav Artzi},
      year={2026},
      eprint={2610.19247},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.19247},
}
```
