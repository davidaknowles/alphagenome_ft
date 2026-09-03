#!/bin/bash
#SBATCH --job-name=zemke24_atac_materialize
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zemke2024_atac_materialize_%j.out
#SBATCH --error=logs/zemke2024_atac_materialize_%j.err

set -euo pipefail
source /gpfs/commons/home/daknowles/venv/jax/bin/activate
export LD_LIBRARY_PATH=/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python scripts/allen_atac_reprocessing/materialize.py \
  --shard-dir outputs/v0data/zemke2024-atac-raw/coverage_shards \
  --fai /gpfs/commons/groups/knowles_lab/index/hg38/hg38.fa.fai \
  --output-dir outputs/v0data/zemke2024-atac-raw/bigwigs \
  --field coverage_spmr \
  --head-id zemke2024_all_atac

python scripts/zemke2024_atac_reprocessing/compose_raw_atac_targets.py \
  --gene-targets outputs/v0data/zemke2024-gene-only/targets.json \
  --raw-atac-targets outputs/v0data/zemke2024-atac-raw/bigwigs/targets.json \
  --output outputs/v0data/zemke2024-atac-raw/targets.json
