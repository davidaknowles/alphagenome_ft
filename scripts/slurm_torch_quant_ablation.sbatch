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

if command -v module >/dev/null 2>&1; then
  module load Ninja/1.12.1-GCCcore-14.2.0 || true
fi

PYTHON="${PYTHON:-$HOME/venv/torchfix/bin/python}"
STRATEGY_FILE="${STRATEGY_FILE:?Set STRATEGY_FILE.}"
BATCH_SIZE_FILE="${BATCH_SIZE_FILE:-}"
N_BATCH_SIZES="${N_BATCH_SIZES:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"
TORCH_WEIGHTS="${TORCH_WEIGHTS:?Set TORCH_WEIGHTS.}"
TORCH_REPO="${TORCH_REPO:-/gpfs/commons/home/daknowles/projects/alphagenome-pytorch}"
TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

mapfile -t STRATEGIES < "$STRATEGY_FILE"
if [[ -n "$BATCH_SIZE_FILE" ]]; then
  mapfile -t BATCH_SIZES < "$BATCH_SIZE_FILE"
else
  BATCH_SIZES=("${BATCH_SIZE:-8}")
fi

STRATEGY_INDEX=$((SLURM_ARRAY_TASK_ID / N_BATCH_SIZES))
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID % N_BATCH_SIZES))
STRATEGY="${STRATEGIES[$STRATEGY_INDEX]}"
BATCH_SIZE_VALUE="${BATCH_SIZES[$BATCH_INDEX]}"
RUN_DIR="$OUTPUT_ROOT/torch_${STRATEGY}_batch${BATCH_SIZE_VALUE}"

mkdir -p "$RUN_DIR" logs/quant_ablation

ARGS=(
  scripts/run_quant_ablation.py
  --backend torch
  --strategy "$STRATEGY"
  --torch-repo "$TORCH_REPO"
  --torch-weights "$TORCH_WEIGHTS"
  --output-dir "$RUN_DIR"
  --splits "${SPLITS:-valid,test}"
  --batch-size "$BATCH_SIZE_VALUE"
  --target-workers "${TARGET_WORKERS:-12}"
  --window-workers "${WINDOW_WORKERS:-4}"
)

if [[ -n "$TARGET_CACHE_DIR" ]]; then
  ARGS+=(--target-cache-dir "$TARGET_CACHE_DIR")
fi
if [[ -n "${MAX_BATCHES:-}" ]]; then
  ARGS+=(--max-batches "$MAX_BATCHES")
fi

"$PYTHON" "${ARGS[@]}"
