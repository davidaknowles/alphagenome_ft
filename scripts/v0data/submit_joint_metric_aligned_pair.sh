#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data outputs/v0data/joint-objective-variants

sbatch_bin="${SBATCH_BIN:-sbatch}"
num_epochs="${NUM_EPOCHS:-100}"
target_workers="${TARGET_WORKERS:-12}"
window_workers="${WINDOW_WORKERS:-4}"
source scripts/v0data/joint_continuation_lib.sh

source_run="${SOURCE_RUN:-checkpoints/v0data/joint_all_nonencode_lora_locon_lr1e4_reset}"
source_epoch="${SOURCE_EPOCH:-22}"
source_checkpoint="${SOURCE_CHECKPOINT:-${source_run}/best}"
learning_rate="${LEARNING_RATE:-1e-4}"
snapshot="checkpoints/v0data/joint_all_nonencode_lora_locon_metric_aligned_seed_epoch${source_epoch}"
canonical_config="$(realpath outputs/v0data/joint-all-nonencode/datasets.json)"
metric_root="outputs/v0data/joint-objective-variants/metric-aligned"

"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_joint_metric_aligned_config.py \
  --input "$canonical_config" \
  --output-dir "$metric_root"
metric_config="$(realpath "$metric_root/datasets.json")"

snapshot_checkpoint "$source_checkpoint" "$source_epoch" "$snapshot"
test -f "$canonical_config"
test -f "$metric_config"

submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  _lr1e4_epoch22_reset_control "$canonical_config"
submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  _lr1e4_epoch22_metric_aligned_reset "$metric_config"
