#!/bin/bash
#SBATCH --job-name=hda_rna_reliability
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v0data/hda_rna_reliability-%j.out
#SBATCH --error=logs/v0data/hda_rna_reliability-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data results
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

"${HOME}/venv/jax/bin/python" scripts/v0data/hda_rna/audit_count_sampling_reliability.py \
  --rna-h5ad /gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/anndata/rna_pseudobulk.h5ad \
  --gene-supervision outputs/v0data/hda-joint/gene_expression_supervision.npz \
  --repeats 20 \
  --json-output results/v0data_hda_rna_count_sampling_reliability.json \
  --markdown-output results/v0data_hda_rna_count_sampling_reliability.md
