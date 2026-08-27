#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data

sbatch_bin="${SBATCH_BIN:-sbatch}"
source scripts/v0data/joint_continuation_lib.sh

source_run="${SOURCE_RUN:-checkpoints/v0data/joint_all_nonencode_lora_locon_lr3e5_epoch31_metric_tempered_reset}"
source_epoch="${SOURCE_EPOCH:-32}"
source_checkpoint="${SOURCE_CHECKPOINT:-${source_run}/best}"
snapshot="checkpoints/v0data/joint_all_nonencode_lora_locon_source_balance_seed_epoch${source_epoch}"
config_dir="outputs/v0data/joint-objective-variants/metric-tempered-source-balanced"
"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_joint_metric_aligned_config.py \
  --input outputs/v0data/joint-all-nonencode/datasets.json \
  --output-dir "$config_dir" \
  --zemke-weight 3 \
  --sampling-strategy equal_sources
dataset_config="$(realpath "${config_dir}/datasets.json")"
learning_rate="${LEARNING_RATE:-1e-4}"
num_epochs="${NUM_EPOCHS:-100}"
target_workers="${TARGET_WORKERS:-12}"
window_workers="${WINDOW_WORKERS:-4}"
run_suffix="${RUN_SUFFIX:-_source_balanced_epoch32_reset}"
initial_dependency="${INITIAL_DEPENDENCY:-}"

snapshot_checkpoint "$source_checkpoint" "$source_epoch" "$snapshot"
test "$(jq -er '.sampling_strategy' "$dataset_config")" = equal_sources

submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  "$run_suffix" "$dataset_config"
