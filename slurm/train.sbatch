#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@300
#SBATCH --requeue
#SBATCH --job-name=cplm
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
# One rank per GPU. From the repo root:
#   sbatch -A <account> [--nodes=N] slurm/train.sbatch configs/<config>.json
# Extra srun options (e.g. a container or uenv) go in $SRUN_ARGS.
#
# Time limit: 5 min before it, SLURM sends SIGUSR1 to the training tasks, whose
# default action kills them at once (a Python handler would only run once the
# main thread leaves CUDA/NCCL calls, often too late). srun then exits with 138
# (128 + SIGUSR1) and this script requeues the job, which resumes from the
# newest checkpoint. Any other failure ends the job.
mkdir -p logs
export CONFIG="${1:?usage: sbatch slurm/train.sbatch <config.json>}"
srun ${SRUN_ARGS:-} slurm/task.sh
rc=$?
if [ "$rc" -eq 138 ]; then
    echo "[requeue] time limit near, requeueing $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$rc"
