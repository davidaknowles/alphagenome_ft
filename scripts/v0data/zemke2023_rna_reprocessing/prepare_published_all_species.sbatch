#!/bin/bash
#SBATCH --job-name=zemke23_published_genes
#SBATCH --partition=io
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/v0data/zemke23_published_genes-%A_%a.out
#SBATCH --error=logs/v0data/zemke23_published_genes-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

species=(human macaque marmoset mouse)
index="${SLURM_ARRAY_TASK_ID}"
name="${species[$index]}"

"${HOME}/venv/jax/bin/python" scripts/v0data/zemke2023_rna_reprocessing/prepare_published_gene_supervision.py \
  --source-supervision "outputs/v0data/zemke2023-gene-supervision/$name/gene_expression_supervision.npz" \
  --source-targets "outputs/v0data/zemke2023-species/$name/targets.json" \
  --output-dir "outputs/v0data/zemke2023-published-gene-supervision/$name" \
  --species "$name" \
  --gene-loss-weight 1 \
  --coverage-loss-weight 1 \
  --correlation-loss-weight 10
