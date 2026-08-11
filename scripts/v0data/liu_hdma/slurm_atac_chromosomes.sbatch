#!/bin/bash
#SBATCH --job-name=liu_atac_chrom
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-23%8
#SBATCH --output=logs/v0data/liu_atac_chrom-%A_%a.out
#SBATCH --error=logs/v0data/liu_atac_chrom-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/liu-hdma/atac_shards
module load Python/3.12.3-GCCcore-13.3.0
module load HTSlib/1.24-GCC-14.2.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

chromosomes=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY)
chromosome="${chromosomes[$SLURM_ARRAY_TASK_ID]}"
fai=/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa.fai
chromosome_size="$(awk -v chromosome="$chromosome" '$1 == chromosome {print $2}' "$fai")"
test -n "$chromosome_size"
root=/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/liu2026Multiomics

"${HOME}/venv/jax/bin/python" scripts/v0data/liu_hdma/aggregate_atac_chromosome.py \
  --metadata outputs/v0data/liu2026/source/per_cell_meta.csv \
  --totals outputs/v0data/liu-hdma/atac_totals.npz \
  --fragment-root "$root/sample_atac_fragments" \
  --chromosome "$chromosome" \
  --chromosome-size "$chromosome_size" \
  --output "outputs/v0data/liu-hdma/atac_shards/${chromosome}_100bp.npz"
