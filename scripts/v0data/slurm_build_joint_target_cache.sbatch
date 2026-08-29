#!/bin/bash
#SBATCH --job-name=v0_joint_cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/v0data/v0_joint_cache-%j.out
#SBATCH --error=logs/v0data/v0_joint_cache-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4

dataset_config="${DATASET_CONFIG:?DATASET_CONFIG is required}"
target_cache_dir="${TARGET_CACHE_DIR:?TARGET_CACHE_DIR is required}"
cache_workers="${TARGET_CACHE_WORKERS:-24}"
extra_args=()
for split in train valid test; do
  variable="LIMIT_${split^^}"
  limit="${!variable:-}"
  if [[ -n "$limit" ]]; then
    extra_args+=("--limit-${split}" "$limit")
  fi
done
if [[ "${OVERWRITE_TARGET_CACHE:-0}" == "1" ]]; then
  extra_args+=(--overwrite-target-cache)
fi

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --dataset-config "$dataset_config" \
  --checkpoint-dir "${CHECKPOINT_DIR:-/scratch/daknowles/v0data-cache-build}" \
  --target-cache-dir "$target_cache_dir" \
  --target-cache-dtype "${TARGET_CACHE_DTYPE:-float16}" \
  --target-cache-workers "$cache_workers" \
  --window-size "${WINDOW_SIZE:-131072}" \
  --stride "${STRIDE:-131072}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --target-workers 0 \
  --window-workers 0 \
  --build-target-cache \
  --build-target-cache-only \
  "${extra_args[@]}"
