#!/bin/bash
#SBATCH --job-name=zemke23_rna_rel_collate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/v0data/zemke23_rna_rel_collate-%j.out
#SBATCH --error=logs/v0data/zemke23_rna_rel_collate-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data results

"${HOME}/venv/jax/bin/python" scripts/v0data/johansen_rna_reprocessing/collate_donor_reliability.py \
  results/v0data_zemke2023_human_rna_donor_reliability.json \
  results/v0data_zemke2023_macaque_rna_donor_reliability.json \
  results/v0data_zemke2023_marmoset_rna_donor_reliability.json \
  results/v0data_zemke2023_mouse_rna_donor_reliability.json \
  --expected-species human,macaque,marmoset,mouse \
  --title "Zemke 2023 RNA donor reliability" \
  --json-output results/v0data_zemke2023_rna_donor_reliability.json \
  --markdown-output results/v0data_zemke2023_rna_donor_reliability.md
