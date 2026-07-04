#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/venv/torchfix/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/og_low_vram/$(date +%Y%m%d_%H%M%S)_parity_driver}"
OG_WEIGHTS="${OG_WEIGHTS:-/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors}"
BF16_WEFF_WEIGHTS="${BF16_WEFF_WEIGHTS:-$ROOT/outputs/og_low_vram/alphagenome_og_bf16_weff.safetensors}"
TORCH_REPO="${TORCH_REPO:-/gpfs/commons/home/daknowles/projects/alphagenome-pytorch}"

mkdir -p "$OUTPUT_ROOT" logs/og_low_vram

if [[ ! -s "$BF16_WEFF_WEIGHTS" ]]; then
  PREP_JOB=$(sbatch --wait \
    -J ag-og-weff \
    -p cpu \
    --cpus-per-task=4 \
    --mem=32G \
    --time=4:00:00 \
    -o logs/og_low_vram/%x_%j.out \
    -e logs/og_low_vram/%x_%j.err \
    --wrap "cd '$ROOT' && '$PYTHON' scripts/prepare_og_bf16_weff_checkpoint.py --og-weights '$OG_WEIGHTS' --output '$BF16_WEFF_WEIGHTS' --torch-repo '$TORCH_REPO'" | awk '{print $4}')
  echo "prep_job_id=$PREP_JOB"
else
  echo "prep_job_id=already_exists"
fi

export ROOT PYTHON OUTPUT_ROOT OG_WEIGHTS BF16_WEFF_WEIGHTS TORCH_REPO
export CHROM="${CHROM:-chr9}"
export HEAD="${HEAD:-atac}"
export MAX_WINDOWS="${MAX_WINDOWS:-}"

DRIVER_JOB=$(sbatch scripts/slurm_og_low_vram_parity_driver.sbatch | awk '{print $4}')

echo "driver_job_id=$DRIVER_JOB"
echo "output_root=$OUTPUT_ROOT"
echo "bf16_weff_weights=$BF16_WEFF_WEIGHTS"
