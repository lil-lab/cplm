"""Pretrain a CPLM from scratch with Lightning Fabric.

    python -m cplm.train configs/cplm_135M_sw256.json

Run id = $RUN_ID, else $SLURM_JOB_ID. Checkpoints go to <ckpt_root>/<run id>/;
a run restarted with the same id (SLURM requeue, or RUN_ID=<old id>) resumes
from its newest complete checkpoint and appends to the same wandb run. Requeueing
before the time limit is done by slurm/train.sbatch, not here.
"""

import glob
import itertools
import json
import math
import os
import re
import shutil
import signal
import sys

import lightning as L
import torch
from lightning.fabric.strategies import DDPStrategy, FSDPStrategy
from torch.distributed.fsdp import MixedPrecision

from .data import build_loaders, prepare_batch
from .model import CPLM, build_backbone


def wsd_lambda(warmup, total, cooldown, min_ratio):
    """Warmup-stable-decay: linear warmup, flat, cosine cooldown to min_ratio."""
    def f(step):
        if step < warmup:
            return min_ratio + (1 - min_ratio) * step / warmup
        if step < total - cooldown:
            return 1.0
        if step < total:
            progress = (step - (total - cooldown)) / cooldown
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        return min_ratio
    return f


def build_strategy(name, decoder_layer_cls, reduce_dtype):
    if name == "ddp":
        return DDPStrategy(gradient_as_bucket_view=True, broadcast_buffers=False)
    # FSDP wraps only the decoder layers: the tied embedding / LM head and the copy
    # pointer stay whole in the root unit, which the vocab-tiled loss slices directly.
    return FSDPStrategy(
        auto_wrap_policy={decoder_layer_cls},
        sharding_strategy={"fsdp": "SHARD_GRAD_OP", "fsdp_full": "FULL_SHARD"}[name],
        state_dict_type="sharded",
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                       reduce_dtype=getattr(torch, reduce_dtype),
                                       buffer_dtype=torch.bfloat16),
    )


class Wandb:
    def __init__(self, fabric, project, name, run_id, config):
        self.run = None
        if fabric.global_rank == 0 and project:
            import wandb
            self.run = wandb.init(project=project, name=name, id=run_id, resume="allow",
                                  config=config)

    def log(self, metrics, step):
        if self.run is not None:
            self.run.log(metrics, step=step)


def checkpoint_step(path):
    m = re.search(r"(\d+)\.ckpt$", path)
    return int(m.group(1)) if m else -1


def latest_checkpoint(ckpt_dir):
    """Newest checkpoint that was fully written (a sharded save ends with meta.pt)."""
    done = [p for p in glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
            if os.path.exists(os.path.join(p, "meta.pt")) or
            (os.path.isfile(p) and os.path.getsize(p) > 0)]
    return max(done, key=checkpoint_step, default=None)


def main(cfg):
    m, o, t = cfg["model"], cfg["optim"], cfg["train"]
    torch.set_float32_matmul_precision("medium")
    run_id = os.environ.get("RUN_ID") or os.environ.get("SLURM_JOB_ID") or "local"

    backbone = build_backbone(m)
    model = CPLM(backbone, copy_id=m["vocab_size"], compile_loss=t["compile"]).float()
    decoder_layer_cls = type(backbone.model.layers[0])

    fabric = L.Fabric(accelerator="cuda", devices="auto",
                      num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
                      strategy=build_strategy(t["strategy"], decoder_layer_cls,
                                              t.get("reduce_dtype", "float32")),
                      precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(t["seed"])

    per_step = t["micro_batch_size"] * fabric.world_size
    assert t["batch_size"] % per_step == 0, "batch_size must be a multiple of micro_batch_size * world_size"
    accum = t["batch_size"] // per_step

    if t["compile"]:
        model = torch.compile(model)   # Fabric re-applies it after DDP/FSDP wrapping
    optimizer = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"],
                                  betas=tuple(o["betas"]), eps=o["eps"])
    model, optimizer = fabric.setup(model, optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, wsd_lambda(o["warmup_steps"], t["steps"], o["cooldown_steps"], o["min_lr_ratio"]))

    num_workers = t["num_workers"]
    train_loader, val_loader = build_loaders(t["data_path"], t["seq_length"],
                                             t["micro_batch_size"], num_workers)
    train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)
    val_loader = fabric.setup_dataloaders(val_loader, use_distributed_sampler=False)

    ckpt_dir = os.path.join(t["ckpt_root"], run_id)
    step = 0

    def save(name):
        state = {"model": model, "optimizer": optimizer, "step": step,
                 "scheduler": scheduler.state_dict(),
                 "dataloader": train_loader._dataloader.state_dict(),
                 "world_size": fabric.world_size, "num_workers": num_workers}
        fabric.save(os.path.join(ckpt_dir, f"{name}.ckpt"), state)
        fabric.print(f"[ckpt] step {step} -> {name}.ckpt")
        if name.startswith("last") and fabric.global_rank == 0:   # keep only the newest last-*
            for old in glob.glob(os.path.join(ckpt_dir, "last-*.ckpt")):
                if checkpoint_step(old) < step:
                    shutil.rmtree(old) if os.path.isdir(old) else os.remove(old)
        fabric.barrier()

    resume = latest_checkpoint(ckpt_dir) if os.path.isdir(ckpt_dir) else None
    if resume:
        rest = fabric.load(resume, {"model": model, "optimizer": optimizer}, weights_only=False)
        assert (rest["world_size"], rest["num_workers"]) == (fabric.world_size, num_workers), \
            "litdata resume state is per rank and per worker: keep world size and num_workers"
        scheduler.load_state_dict(rest["scheduler"])
        train_loader._dataloader.load_state_dict(rest["dataloader"])
        step = rest["step"]
        fabric.print(f"[ckpt] resumed step {step} <- {resume}")

    logger = Wandb(fabric, t.get("wandb_project"), cfg["name"], run_id, cfg)
    eos = m["eos_id"]

    def forward(batch):
        input_ids, labels, pos, doc_ids = prepare_batch(batch, eos)
        return model(input_ids, labels, pos, doc_ids)

    @torch.no_grad()
    def validate():
        model.eval()
        losses = [forward(b)["loss"] for b in itertools.islice(val_loader, t["val_batches"])]
        loss = fabric.all_reduce(torch.stack(losses).mean(), reduce_op="mean").item()
        logger.log({"val/lm_loss": loss}, step)
        fabric.print(f"[val] step {step} loss {loss:.4f}")
        model.train()

    if step == 0:
        validate()
    model.train()
    data = iter(train_loader)
    while step < t["steps"]:
        loss = copy_rate = sink_rate = 0.0
        for micro in range(accum):
            with fabric.no_backward_sync(model, enabled=micro < accum - 1):
                out = forward(next(data))
                fabric.backward(out["loss"] / accum)
            loss += out["loss"].item() / accum
            copy_rate += out["copy_rate"].item() / accum
            sink_rate += out["sink_rate"].item() / accum
        fabric.clip_gradients(model, optimizer, max_norm=o["grad_clip"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        # torch.compile installs a C-level SIGUSR1 handler that swallows the signal
        # slurm/train.sbatch relies on to stop the job before its time limit.
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)

        if step % t["log_every"] == 0:
            logger.log({"train/lm_loss": loss, "train/copy_rate": copy_rate,
                        "train/sink_rate": sink_rate, "train/lr": scheduler.get_last_lr()[0]}, step)
            fabric.print(f"step {step} loss {loss:.4f} copy {copy_rate:.3f} sink {sink_rate:.3f}")
        if step % t["val_every"] == 0:
            validate()
        if step % t["ckpt_every"] == 0 or step == t["steps"]:
            save(str(step))
        elif step % t["last_ckpt_every"] == 0:
            save(f"last-{step}")
    fabric.print("[train] done")


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        main(json.load(f))
