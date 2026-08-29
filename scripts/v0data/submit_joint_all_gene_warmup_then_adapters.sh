#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python_bin="${HOME}/venv/jax/bin/python"
zemke2023_dir="outputs/v0data/zemke2023-gene-only-species"
zemke2024_target="outputs/v0data/zemke2024-gene-only/targets.json"
base_dir="outputs/v0data/joint-all-nonencode-all-gene"
config_dir="outputs/v0data/joint-objective-variants/metric-tempered-all-gene"
source_specific_heads="${SOURCE_SPECIFIC_HEADS:-0}"
separate_head_updates="${SEPARATE_HEAD_UPDATES:-0}"
if [[ "$source_specific_heads" != "0" && "$source_specific_heads" != "1" ]]; then
  printf 'SOURCE_SPECIFIC_HEADS must be 0 or 1, got %s.\n' "$source_specific_heads" >&2
  exit 2
fi
if [[ "$separate_head_updates" != "0" && "$separate_head_updates" != "1" ]]; then
  printf 'SEPARATE_HEAD_UPDATES must be 0 or 1, got %s.\n' "$separate_head_updates" >&2
  exit 2
fi
variant_suffix=""
if [[ "$separate_head_updates" == "1" ]]; then
  variant_suffix="-separate-heads"
  config_dir="${config_dir}${variant_suffix}"
fi
if [[ "$source_specific_heads" == "1" ]]; then
  config_dir="outputs/v0data/joint-objective-variants/metric-tempered-all-gene-source-balanced${variant_suffix}"
fi

"$python_bin" scripts/v0data/zemke2023_rna_reprocessing/prepare_gene_only_species.py \
  --input outputs/v0data/zemke2023-species/species.json \
  --supervision-root outputs/v0data/zemke2023-gene-supervision \
  --output-dir "$zemke2023_dir" \
  --correlation-loss-weight 1
"$python_bin" scripts/v0data/zemke2024_rna_reprocessing/prepare_gene_only_target.py \
  --input outputs/v0data/zemke2024-gene-supervision/targets.json \
  --supervision outputs/v0data/zemke2024-gene-supervision/gene_expression_supervision.npz \
  --output "$zemke2024_target" \
  --correlation-loss-weight 1
"$python_bin" scripts/v0data/prepare_joint_dataset_config.py \
  --zemke2023-species "${zemke2023_dir}/species.json" \
  --zemke2024-targets "$zemke2024_target" \
  --output "${base_dir}/datasets.json"
metric_args=(
  --input "${base_dir}/datasets.json"
  --output-dir "$config_dir"
  --zemke-weight 3
  --zemke-rna-weight 1
  --zemke2024-rna-weight 1
)
if [[ "$source_specific_heads" == "1" ]]; then
  metric_args+=(--sampling-strategy equal_sources)
fi
if [[ "$separate_head_updates" == "1" ]]; then
  metric_args+=(--head-update-strategy separate_heads)
fi
"$python_bin" scripts/v0data/prepare_joint_metric_aligned_config.py \
  "${metric_args[@]}"

dataset_config="${config_dir}/datasets.json"
run_tag="all_gene"
if [[ "$source_specific_heads" == "1" ]]; then
  source_specific_dir="outputs/v0data/joint-objective-variants/metric-tempered-all-gene-source-specific${variant_suffix}"
  "$python_bin" scripts/v0data/prepare_source_specific_joint_heads.py \
    --input "$dataset_config" \
    --output-dir "$source_specific_dir"
  dataset_config="${source_specific_dir}/datasets.json"
  run_tag="all_gene_source_specific"
fi
if [[ "$separate_head_updates" == "1" ]]; then
  run_tag="${run_tag}_separate_heads"
fi
target_cache_dir=""
if [[ "$source_specific_heads" == "1" ]]; then
  cache_variant=""
  if [[ "$separate_head_updates" == "1" ]]; then
    cache_variant="-separate-heads"
  fi
  target_cache_dir="${TARGET_CACHE_DIR:-outputs/v0data/target-caches/joint-source-specific${cache_variant}-all-gene-valid-test-f16}"
fi

DATASET_CONFIG="$dataset_config" \
RUN_TAG="$run_tag" \
TARGET_CACHE_DIR="$target_cache_dir" \
TARGET_CACHE_SPLITS="${TARGET_CACHE_SPLITS:-valid;test}" \
SMOKE_LIMIT_TRAIN="${SMOKE_LIMIT_TRAIN:-40}" \
SMOKE_LIMIT_VALID="${SMOKE_LIMIT_VALID:-40}" \
SMOKE_LIMIT_TEST="${SMOKE_LIMIT_TEST:-40}" \
bash scripts/v0data/submit_joint_head_warmup_then_adapters.sh
