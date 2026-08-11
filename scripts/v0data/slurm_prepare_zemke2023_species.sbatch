#!/bin/bash
#SBATCH --job-name=zemke23_refs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v0data/zemke23_refs-%j.out
#SBATCH --error=logs/v0data/zemke23_refs-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/zemke2023-species
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_zemke2023_species.py \
  --human-fasta /gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa \
  --macaque-fasta outputs/allen_brain_multiome_multispecies_v1/references/macaque.fa \
  --targets-root outputs/v0data \
  --output-dir outputs/v0data/zemke2023-species
