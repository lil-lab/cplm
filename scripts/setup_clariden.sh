#!/bin/bash
# Build .venv on CSCS Clariden (GH200, aarch64). Run inside the uenv so the Python
# has headers and nvcc/gcc are available for torch.compile:
#   srun -A <account> --uenv=pytorch/v2.9.1:v2 --view=default bash scripts/setup_clariden.sh
set -euo pipefail
cd "$(dirname "$0")/.."
/user-environment/env/default/bin/python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch==2.12.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[data,test]"
# Prebuilt FlashAttention-3 wheel for aarch64 / cu130 / torch 2.12 (no source build).
pip install "https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.05.20-072445f/flash_attn_3-3.0.0%2B20260520.cu130torch2120cxx11abitrue.417891-cp39-abi3-linux_aarch64.whl"
