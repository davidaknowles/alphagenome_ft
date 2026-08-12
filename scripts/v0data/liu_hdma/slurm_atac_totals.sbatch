#!/bin/bash
#SBATCH --job-name=liu_atac_totals
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v0data/liu_atac_totals-%j.out
#SBATCH --error=logs/v0data/liu_atac_totals-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/liu-hdma
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

root=/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/liu2026Multiomics
"${HOME}/venv/jax/bin/python" scripts/v0data/liu_hdma/compute_atac_totals.py \
  --metadata outputs/v0data/liu2026/source/per_cell_meta.csv \
  --cluster-manifest outputs/v0data/liu-hdma/clusters.json \
  --fragment-root "$root/sample_atac_fragments" \
  --output outputs/v0data/liu-hdma/atac_totals.npz
