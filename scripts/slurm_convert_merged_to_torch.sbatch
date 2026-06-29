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
PYTHON="${PYTHON:-$HOME/venv/torch/bin/python}"
JAX_CHECKPOINT="${JAX_CHECKPOINT:?Set JAX_CHECKPOINT to merged_jax/checkpoint.}"
TORCH_WEIGHTS="${TORCH_WEIGHTS:?Set TORCH_WEIGHTS to output .safetensors path.}"

mkdir -p "$(dirname "$TORCH_WEIGHTS")" "$ROOT/logs/quant_ablation"
cd "$TORCH_REPO"
"$PYTHON" scripts/convert_weights.py "$JAX_CHECKPOINT" --output "$TORCH_WEIGHTS" --safetensors
