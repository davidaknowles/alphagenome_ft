#!/bin/bash
#SBATCH --job-name=zemke23_gene_audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v0data/zemke23_gene_audit-%j.out
#SBATCH --error=logs/v0data/zemke23_gene_audit-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

"${HOME}/venv/jax/bin/python" scripts/v0data/zemke2023_rna_reprocessing/audit_gene_track_agreement.py \
  --root outputs/v0data/zemke2023-gene-supervision \
  --json-output results/v0data_zemke2023_gene_target_agreement.json \
  --markdown-output results/v0data_zemke2023_gene_target_agreement.md
