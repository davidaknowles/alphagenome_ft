#!/usr/bin/env bash
#SBATCH -J ag-quant-convert
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH -o logs/quant_ablation/%x_%j.out
#SBATCH -e logs/quant_ablation/%x_%j.err

set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
TORCH_REPO="${TORCH_REPO:-/gpfs/commons/home/daknowles/projects/alphagenome-pytorch}"
PYTHON="${PYTHON:-$HOME/venv/torchfix/bin/python}"
JAX_CHECKPOINT="${JAX_CHECKPOINT:?Set JAX_CHECKPOINT to merged_jax/checkpoint.}"
TORCH_WEIGHTS="${TORCH_WEIGHTS:?Set TORCH_WEIGHTS to output .pth path.}"
BASE_TORCH_WEIGHTS="${BASE_TORCH_WEIGHTS:-${TORCH_WEIGHTS%.*}.base.safetensors}"
BIGWIG_DIR="${BIGWIG_DIR:-/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs}"
HEAD_ID="${HEAD_ID:-humanbraindev_atac}"
ASSAY_TYPE="${ASSAY_TYPE:-atac}"

mkdir -p "$(dirname "$TORCH_WEIGHTS")" "$ROOT/logs/quant_ablation"
cd "$TORCH_REPO"
"$PYTHON" scripts/convert_weights.py "$JAX_CHECKPOINT" --output "$BASE_TORCH_WEIGHTS" --safetensors
cd "$ROOT"
"$PYTHON" scripts/convert_merged_jax_to_torch_full.py \
  --jax-checkpoint "$JAX_CHECKPOINT" \
  --base-torch-weights "$BASE_TORCH_WEIGHTS" \
  --output "$TORCH_WEIGHTS" \
  --torch-repo "$TORCH_REPO" \
  --head-id "$HEAD_ID" \
  --assay-type "$ASSAY_TYPE" \
  --bigwig-dir "$BIGWIG_DIR"
