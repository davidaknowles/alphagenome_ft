#!/bin/bash
#SBATCH --job-name=zemke24_gene_smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v0data/zemke24_gene_smoke-%j.out
#SBATCH --error=logs/v0data/zemke24_gene_smoke-%j.err

set -euo pipefail
cd "$(dirname "$0")/../../.."
source ~/venv/jax/bin/activate

python scripts/v0data/zemke2024_rna_reprocessing/smoke_donor_aggregation.py \
  --donor hc11 \
  --matrix /gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/zemke2024Epigenetic/RNAmatrix/hc11/GSE278576_hc11_raw_feature_bc_matrix.h5 \
  --metadata outputs/v0data/zemke2024-gene-supervision/source/GSE278576_hippocampus_RNA_seurat_object_filtered_cells_metadata.tsv.gz \
  --targets outputs/v0data/zemke2024-all/targets.json \
  --output results/v0data_zemke2024_hc11_raw_count_smoke.json
