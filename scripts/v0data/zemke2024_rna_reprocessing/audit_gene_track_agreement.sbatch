#!/bin/bash
#SBATCH --job-name=zemke24_gene_audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v0data/zemke24_gene_audit-%j.out
#SBATCH --error=logs/v0data/zemke24_gene_audit-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate

python scripts/v0data/zemke2024_rna_reprocessing/audit_gene_track_agreement.py \
  --output-dir outputs/v0data/zemke2024-gene-supervision \
  --json-output results/v0data_zemke2024_gene_track_agreement.json \
  --markdown-output results/v0data_zemke2024_gene_track_agreement.md
