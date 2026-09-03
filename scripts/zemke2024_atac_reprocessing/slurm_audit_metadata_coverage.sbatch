#!/bin/bash
#SBATCH --job-name=zemke24_atac_metadata
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/zemke2024_atac_metadata_%j.out
#SBATCH --error=logs/zemke2024_atac_metadata_%j.err

set -euo pipefail
source /gpfs/commons/home/daknowles/venv/jax/bin/activate
export LD_LIBRARY_PATH=/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python scripts/zemke2024_atac_reprocessing/audit_metadata_coverage.py \
  --fragment-root /gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/zemke2024Epigenetic/ATACfragments \
  --metadata outputs/v0data/zemke2024-gene-supervision/source/GSE278576_hippocampus_RNA_seurat_object_filtered_cells_metadata.tsv.gz \
  --output results/v0data_zemke2024_atac_metadata_coverage.json
