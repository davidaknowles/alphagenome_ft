#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python_bin="${HOME}/venv/jax/bin/python"
zemke2023_dir="outputs/v0data/zemke2023-gene-only-species"
zemke2024_target="outputs/v0data/zemke2024-gene-only/targets.json"
base_dir="outputs/v0data/joint-all-nonencode-all-gene"
config_dir="outputs/v0data/joint-objective-variants/metric-tempered-all-gene"

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
"$python_bin" scripts/v0data/prepare_joint_metric_aligned_config.py \
  --input "${base_dir}/datasets.json" \
  --output-dir "$config_dir" \
  --zemke-weight 3 \
  --zemke-rna-weight 1 \
  --zemke2024-rna-weight 1

DATASET_CONFIG="${config_dir}/datasets.json" \
RUN_TAG=all_gene \
SMOKE_LIMIT_TRAIN="${SMOKE_LIMIT_TRAIN:-40}" \
SMOKE_LIMIT_VALID="${SMOKE_LIMIT_VALID:-40}" \
SMOKE_LIMIT_TEST="${SMOKE_LIMIT_TEST:-40}" \
bash scripts/v0data/submit_joint_head_warmup_then_adapters.sh
