#!/bin/bash
#SBATCH --job-name=liu_rna_reduce
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v0data/liu_rna_reduce-%j.out
#SBATCH --error=logs/v0data/liu_rna_reduce-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/liu-hdma
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

"${HOME}/venv/jax/bin/python" scripts/v0data/liu_hdma/reduce_rna.py \
  --metadata outputs/v0data/liu2026/source/per_cell_meta.csv \
  --cluster-manifest outputs/v0data/liu-hdma/clusters.json \
  --sample-dir outputs/v0data/liu-hdma/rna_samples \
  --output outputs/v0data/liu-hdma/rna_pseudobulk.h5ad
