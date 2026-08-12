#!/bin/bash
#SBATCH --job-name=head_init_select
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/v0data/head_init_select-%j.out
#SBATCH --error=logs/v0data/head_init_select-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

"${HOME}/venv/jax/bin/python" scripts/v0data/select_pretrained_head_initialization.py \
  --minimum-improvement "${MINIMUM_IMPROVEMENT:-0}" \
  --maximum-modality-regression "${MAXIMUM_MODALITY_REGRESSION:-0}" \
  --json-output results/v0data_pretrained_head_initialization_selection.json \
  --markdown-output results/v0data_pretrained_head_initialization_selection.md
