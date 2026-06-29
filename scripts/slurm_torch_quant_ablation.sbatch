#!/usr/bin/env bash
#SBATCH -J ag-torch-quant
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH -o logs/quant_ablation/%x_%A_%a.out
#SBATCH -e logs/quant_ablation/%x_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/venv/torch/bin/python}"
STRATEGY_FILE="${STRATEGY_FILE:?Set STRATEGY_FILE.}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"
TORCH_WEIGHTS="${TORCH_WEIGHTS:?Set TORCH_WEIGHTS.}"
TORCH_REPO="${TORCH_REPO:-/gpfs/commons/home/daknowles/projects/alphagenome-pytorch}"
TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-}"

export PYTHONPATH="$HOME/venv/jax/lib/python3.12/site-packages:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

mapfile -t STRATEGIES < "$STRATEGY_FILE"
STRATEGY="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$OUTPUT_ROOT/torch_$STRATEGY"

mkdir -p "$RUN_DIR" logs/quant_ablation

ARGS=(
  scripts/run_quant_ablation.py
  --backend torch
  --strategy "$STRATEGY"
  --torch-repo "$TORCH_REPO"
  --torch-weights "$TORCH_WEIGHTS"
  --output-dir "$RUN_DIR"
  --splits "${SPLITS:-valid,test}"
  --batch-size "${BATCH_SIZE:-8}"
  --target-workers "${TARGET_WORKERS:-12}"
  --window-workers "${WINDOW_WORKERS:-4}"
)

if [[ -n "$TARGET_CACHE_DIR" ]]; then
  ARGS+=(--target-cache-dir "$TARGET_CACHE_DIR")
fi

"$PYTHON" "${ARGS[@]}"
