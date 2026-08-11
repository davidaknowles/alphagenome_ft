#!/bin/bash
#SBATCH -J allen_atac_raw
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/allen_atac_reprocess/%x-%j.out
#SBATCH -e logs/allen_atac_reprocess/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/allen_atac_reprocess outputs/allen_atac_reprocessed

module load HTSlib/1.24-GCC-14.2.0
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/allen_atac_reprocessing/aggregate.py \
  --fragment-root /gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/johansen2025Crossspecies/fragment/human \
  --cell-metadata-h5ad /gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/johansen2025Crossspecies/comprehensive_expression_matrix/Human/Human_HMBA_basalganglia_AIT_pre-print.h5ad \
  --chromosome chr8 \
  --chromosome-size 145138636 \
  --output outputs/allen_atac_reprocessed/chr8_100bp.npz

python scripts/allen_atac_reprocessing/compare.py \
  --shard outputs/allen_atac_reprocessed/chr8_100bp.npz \
  --released-bigwig-dir /gpfs/commons/datasets/controlled/NYGC_AI_Initiative/AllenBrainMultiome/bigwigs/human \
  --starts 5000000,21000000,37000000,53000000,69000000,85000000,101000000,117000000 \
  --output results/allen_atac_reprocessing_chr8.json
