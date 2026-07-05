#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/venv/torchfix/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/og_low_vram/$(date +%Y%m%d_%H%M%S)_parity_driver}"
OG_WEIGHTS="${OG_WEIGHTS:-/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors}"
TORCH_REPO="${TORCH_REPO:-/gpfs/commons/home/daknowles/projects/alphagenome-pytorch}"

mkdir -p "$OUTPUT_ROOT" logs/og_low_vram

export ROOT PYTHON OUTPUT_ROOT OG_WEIGHTS TORCH_REPO
export CHROM="${CHROM:-chr9}"
export HEAD="${HEAD:-atac}"
export MAX_WINDOWS="${MAX_WINDOWS:-}"

DRIVER_JOB=$(sbatch scripts/slurm_og_low_vram_parity_driver.sbatch | awk '{print $4}')

echo "driver_job_id=$DRIVER_JOB"
echo "output_root=$OUTPUT_ROOT"
echo "og_weights=$OG_WEIGHTS"
