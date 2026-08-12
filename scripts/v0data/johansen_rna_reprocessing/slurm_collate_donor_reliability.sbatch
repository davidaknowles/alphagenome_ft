#!/bin/bash
#SBATCH --job-name=johansen_rna_reliability_collate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/v0data/johansen_rna_reliability_collate-%j.out
#SBATCH --error=logs/v0data/johansen_rna_reliability_collate-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4

"${HOME}/venv/jax/bin/python" scripts/v0data/johansen_rna_reprocessing/collate_donor_reliability.py \
  results/v0data_johansen_human_rna_donor_reliability.json \
  results/v0data_johansen_macaque_rna_donor_reliability.json \
  results/v0data_johansen_marmoset_rna_donor_reliability.json \
  --json-output results/v0data_johansen_rna_donor_reliability.json \
  --markdown-output results/v0data_johansen_rna_donor_reliability.md
