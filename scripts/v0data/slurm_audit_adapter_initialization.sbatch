#!/bin/bash
#SBATCH -J adapter_init_audit
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/v0data/%x-%j.out
#SBATCH -e logs/v0data/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"${HOME}/venv/jax/bin/python" scripts/v0data/audit_adapter_initialization.py \
  --targets-config outputs/v0data/hda/targets.json \
  --checkpoint-path /gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1 \
  --output outputs/v0data/adapter_initialization_audit.json
