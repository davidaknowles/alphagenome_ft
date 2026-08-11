#!/bin/bash
#SBATCH -J allen_atac_pileup
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH --array=0-23%12
#SBATCH -o logs/allen_atac_reprocess/%x-%A_%a.out
#SBATCH -e logs/allen_atac_reprocess/%x-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/allen_atac_reprocess outputs/allen_atac_full_depth/shards

module load HTSlib/1.24-GCC-14.2.0
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

chromosomes=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY)
chromosome="${chromosomes[$SLURM_ARRAY_TASK_ID]}"
fai=/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa.fai
chromosome_size="$(awk -v chromosome="$chromosome" '$1 == chromosome {print $2}' "$fai")"
test -n "$chromosome_size"

python scripts/allen_atac_reprocessing/aggregate.py \
  --fragment-root /gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/johansen2025Crossspecies/fragment/human \
  --cell-metadata-h5ad /gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/johansen2025Crossspecies/comprehensive_expression_matrix/Human/Human_HMBA_basalganglia_AIT_pre-print.h5ad \
  --chromosome "$chromosome" \
  --chromosome-size "$chromosome_size" \
  --output "outputs/allen_atac_full_depth/shards/${chromosome}_100bp.npz"
