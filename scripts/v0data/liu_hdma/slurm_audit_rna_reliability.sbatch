#!/bin/bash
#SBATCH --job-name=liu_rna_reliability
#SBATCH --partition=io
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v0data/liu_rna_reliability-%j.out
#SBATCH --error=logs/v0data/liu_rna_reliability-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data results
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

"${HOME}/venv/jax/bin/python" scripts/v0data/liu_hdma/audit_rna_reliability.py \
  --sample-dir outputs/v0data/liu-hdma/rna_samples \
  --gene-supervision outputs/v0data/liu-hdma/joint/gene_expression_supervision.npz \
  --json-output results/v0data_liu_rna_split_half_reliability.json \
  --markdown-output results/v0data_liu_rna_split_half_reliability.md

"${HOME}/venv/jax/bin/python" scripts/v0data/liu_hdma/audit_split_target_rank.py \
  --gene-supervision outputs/v0data/liu-hdma/joint/gene_expression_supervision.npz \
  --chromosomes chr8,chr9 \
  --json-output results/v0data_liu_split_target_rank.json \
  --markdown-output results/v0data_liu_split_target_rank.md
