#!/bin/bash
#SBATCH --job-name=zemke24_atac_raw
#SBATCH --partition=cpu
#SBATCH --array=0-22%4
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/zemke2024_atac_raw_%A_%a.out
#SBATCH --error=logs/zemke2024_atac_raw_%A_%a.err

set -euo pipefail
module load HTSlib/1.24-GCC-14.2.0
source /gpfs/commons/home/daknowles/venv/jax/bin/activate
export LD_LIBRARY_PATH=/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

chromosomes=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)
chromosome=${chromosomes[$SLURM_ARRAY_TASK_ID]}
fai=/gpfs/commons/groups/knowles_lab/index/hg38/hg38.fa.fai
chromosome_size=$(awk -v chromosome="$chromosome" '$1 == chromosome {print $2; exit}' "$fai")
test -n "$chromosome_size"

python scripts/zemke2024_atac_reprocessing/aggregate.py \
  --fragment-root /gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/zemke2024Epigenetic/ATACfragments \
  --metadata outputs/v0data/zemke2024-gene-supervision/source/GSE278576_hippocampus_RNA_seurat_object_filtered_cells_metadata.tsv.gz \
  --chromosome "$chromosome" \
  --chromosome-size "$chromosome_size" \
  --output "outputs/v0data/zemke2024-atac-raw/coverage_shards/${chromosome}.npz"
