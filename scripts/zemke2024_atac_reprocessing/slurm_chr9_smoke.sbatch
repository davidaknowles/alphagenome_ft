#!/bin/bash
#SBATCH --job-name=zemke24_atac_smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/zemke2024_atac_smoke_%j.out
#SBATCH --error=logs/zemke2024_atac_smoke_%j.err

set -euo pipefail
module load HTSlib/1.24-GCC-14.2.0
source /gpfs/commons/home/daknowles/venv/jax/bin/activate
export LD_LIBRARY_PATH=/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python scripts/zemke2024_atac_reprocessing/aggregate.py \
  --fragment-root /gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/zemke2024Epigenetic/ATACfragments \
  --metadata outputs/v0data/zemke2024-gene-supervision/source/GSE278576_hippocampus_RNA_seurat_object_filtered_cells_metadata.tsv.gz \
  --chromosome chr9 \
  --chromosome-size 138394717 \
  --max-donors 1 \
  --output outputs/v0data/zemke2024-atac-reprocessing/chr9_hc11_smoke.npz
