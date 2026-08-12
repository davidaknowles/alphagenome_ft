#!/bin/bash
#SBATCH --job-name=zemke23_rna_reliability
#SBATCH --partition=cpu
#SBATCH --array=0-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v0data/zemke23_rna_reliability-%A_%a.out
#SBATCH --error=logs/v0data/zemke23_rna_reliability-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data results
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

species=(human macaque marmoset mouse)
name=${species[$SLURM_ARRAY_TASK_ID]}
manifest="outputs/v0data/zemke2023-gene-supervision/${name}/manifest.json"

"${HOME}/venv/jax/bin/python" scripts/v0data/zemke2023_rna_reprocessing/audit_donor_reliability.py \
  --species "$name" \
  --manifest "$manifest" \
  --output "results/v0data_zemke2023_${name}_rna_donor_reliability.json"
