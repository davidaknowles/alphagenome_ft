#!/bin/bash
#SBATCH -J ag_hbd_cache
#SBATCH -p gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/humanbraindev_target_cache/%x-%j.out
#SBATCH -e logs/humanbraindev_target_cache/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/humanbraindev_target_cache

source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-daknowles}}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-/scratch/${USER:-daknowles}/alphagenome_fp4/humanbraindev_atac_w${WINDOW_SIZE:-131072}_${TARGET_CACHE_DTYPE:-float16}}"

echo "target_cache_dir=${TARGET_CACHE_DIR}"
echo "target_cache_dtype=${TARGET_CACHE_DTYPE:-float16}"
echo "target_cache_workers=${TARGET_CACHE_WORKERS:-32}"

python scripts/run_humanbraindev_finetune.py \
  --bigwig-dir "${BIGWIG_DIR:-/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs}" \
  --fasta-path "${FASTA_PATH:-/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa}" \
  --checkpoint-dir /tmp/alphagenome_ft_cache_build_unused \
  --split-source "${SPLIT_SOURCE:-chromosome}" \
  --window-size "${WINDOW_SIZE:-131072}" \
  --stride "${STRIDE:-131072}" \
  --valid-chroms "${VALID_CHROMS:-chr8}" \
  --test-chroms "${TEST_CHROMS:-chr9}" \
  --exclude-chroms "${EXCLUDE_CHROMS:-chrM,chrY}" \
  --target-cache-dir "${TARGET_CACHE_DIR}" \
  --target-cache-dtype "${TARGET_CACHE_DTYPE:-float16}" \
  --target-cache-workers "${TARGET_CACHE_WORKERS:-32}" \
  --build-target-cache \
  --build-target-cache-only \
  ${OVERWRITE_TARGET_CACHE:+--overwrite-target-cache} \
  ${LIMIT_TRAIN:+--limit-train "$LIMIT_TRAIN"} \
  ${LIMIT_VALID:+--limit-valid "$LIMIT_VALID"} \
  ${LIMIT_TEST:+--limit-test "$LIMIT_TEST"}
