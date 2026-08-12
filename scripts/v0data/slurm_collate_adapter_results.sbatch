#!/bin/bash
#SBATCH --job-name=v0data_collate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/v0data/v0data_collate-%j.out
#SBATCH --error=logs/v0data/v0data_collate-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data results

"${HOME}/venv/torchfix/bin/python" scripts/v0data/collate_adapter_comparisons.py \
  --json-output results/v0data_adapter_comparisons.json \
  --markdown-output results/v0data_adapter_comparisons.md \
  --variant-json-output results/v0data_objective_screens.json \
  --variant-markdown-output results/v0data_objective_screens.md

"${HOME}/venv/torchfix/bin/python" scripts/v0data/audit_adapter_coverage.py \
  --json-output results/v0data_adapter_coverage.json \
  --markdown-output results/v0data_adapter_coverage.md

"${HOME}/venv/torchfix/bin/python" scripts/v0data/collate_joint_species_evaluations.py \
  --json-output results/v0data_joint_species_evaluations.json \
  --markdown-output results/v0data_joint_species_evaluations.md

echo "Regenerated canonical, screen, coverage, and native-species results."
