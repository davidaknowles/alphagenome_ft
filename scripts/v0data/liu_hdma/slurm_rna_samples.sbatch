#!/bin/bash
#SBATCH --job-name=liu_rna_samples
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-75%8
#SBATCH --output=logs/v0data/liu_rna_samples-%A_%a.out
#SBATCH --error=logs/v0data/liu_rna_samples-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/liu-hdma/rna_samples
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

root=/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/liu2026Multiomics
"${HOME}/venv/jax/bin/python" scripts/v0data/liu_hdma/aggregate_rna_sample.py \
  --metadata outputs/v0data/liu2026/source/per_cell_meta.csv \
  --cluster-manifest outputs/v0data/liu-hdma/clusters.json \
  --expression-root "$root/sample_cell_expression_matrix" \
  --sample-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir outputs/v0data/liu-hdma/rna_samples
