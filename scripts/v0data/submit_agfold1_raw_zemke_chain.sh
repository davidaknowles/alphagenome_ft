#!/bin/bash
# Rerun the selected raw-Zemke source-specific chain (random-init head warmup,
# then LoRA and LoRA+LoCon) from the AlphaGenome FOLD_1 checkpoint, with
# validation and test windows restricted to FOLD_1's held-out folds.
set -euo pipefail

cd "$(dirname "$0")/../.."

dataset_config="outputs/v0data/joint-objective-variants/metric-tempered-all-gene-source-specific-raw-zemke-atac-ag-fold1/datasets.json"
target_cache_dir="outputs/v0data/target-caches/joint-source-specific-raw-zemke-atac-agfold1-valid-test-f32"
export CHECKPOINT_PATH="${HOME}/.cache/kagglehub/models/google/alphagenome/jax/fold_1/1"
test -f "$dataset_config"
test -f "${CHECKPOINT_PATH}/_CHECKPOINT_METADATA"

cache_job=$(sbatch --parsable \
  --export="ALL,DATASET_CONFIG=${dataset_config},TARGET_CACHE_DIR=${target_cache_dir},TARGET_CACHE_DTYPE=float32,TARGET_CACHE_SPLITS=valid;test,TARGET_CACHE_WORKERS=24" \
  scripts/v0data/slurm_build_joint_target_cache.sbatch)
printf 'target-cache=%s\n' "$cache_job"

DATASET_CONFIG="$dataset_config" \
RUN_TAG="all_gene_source_specific_raw_zemke_atac_agfold1" \
PRETRAINED_HEAD_INITIALIZATION=none \
TARGET_CACHE_DIR="$target_cache_dir" \
TARGET_CACHE_SPLITS="valid;test" \
TARGET_CACHE_DTYPE=float32 \
INITIAL_DEPENDENCY="afterok:${cache_job}" \
SMOKE_LIMIT_TRAIN=40 \
SMOKE_LIMIT_VALID=40 \
SMOKE_LIMIT_TEST=40 \
bash scripts/v0data/submit_joint_head_warmup_then_adapters.sh
