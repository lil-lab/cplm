#!/bin/bash
# One SLURM task = one rank. srun runs this script, which execs python so that
# python itself receives SLURM's signals. JIT caches are per rank on node-local
# /tmp (ranks sharing one inductor/triton cache corrupt it on first compile).
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
CACHE="/tmp/${USER}/cplm_jit/${SLURM_JOB_ID:-0}/r${SLURM_PROCID:-0}"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
exec "${PYTHON:-.venv/bin/python}" -m cplm.train "$CONFIG"
