#!/bin/bash
#SBATCH --job-name=zemke24_atac_compare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/zemke2024_atac_compare_%j.out
#SBATCH --error=logs/zemke2024_atac_compare_%j.err

set -euo pipefail
source /gpfs/commons/home/daknowles/venv/jax/bin/activate
export LD_LIBRARY_PATH=/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python scripts/zemke2024_atac_reprocessing/compare_released.py \
  --shard outputs/v0data/zemke2024-atac-reprocessing/chr9_hc11_smoke.npz \
  --targets outputs/v0data/zemke2024-all/targets.json \
  --output results/v0data_zemke2024_atac_chr9_hc11_released_agreement.json
