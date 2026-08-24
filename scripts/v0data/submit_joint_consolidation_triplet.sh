#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data outputs/v0data/joint-objective-variants

sbatch_bin="${SBATCH_BIN:-sbatch}"
num_epochs="${NUM_EPOCHS:-100}"
target_workers="${TARGET_WORKERS:-12}"
window_workers="${WINDOW_WORKERS:-4}"
source scripts/v0data/joint_continuation_lib.sh

source_run="${SOURCE_RUN:-checkpoints/v0data/joint_all_nonencode_lora_locon_lr1e4_epoch22_metric_aligned_reset}"
source_epoch="${SOURCE_EPOCH:-31}"
source_checkpoint="${SOURCE_CHECKPOINT:-${source_run}/best}"
learning_rate="${LEARNING_RATE:-3e-5}"
snapshot="checkpoints/v0data/joint_all_nonencode_lora_locon_consolidation_seed_epoch${source_epoch}"
canonical_config="$(realpath outputs/v0data/joint-all-nonencode/datasets.json)"
aligned_config="$(realpath outputs/v0data/joint-objective-variants/metric-aligned/datasets.json)"
tempered_root="outputs/v0data/joint-objective-variants/metric-tempered"

"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_joint_metric_aligned_config.py \
  --input "$canonical_config" \
  --output-dir "$tempered_root" \
  --zemke-weight 3
tempered_config="$(realpath "$tempered_root/datasets.json")"

snapshot_checkpoint "$source_checkpoint" "$source_epoch" "$snapshot"
test -f "$canonical_config"
test -f "$aligned_config"
test -f "$tempered_config"

submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  _lr3e5_epoch31_canonical_reset "$canonical_config"
submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  _lr3e5_epoch31_metric_aligned_reset "$aligned_config"
submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  _lr3e5_epoch31_metric_tempered_reset "$tempered_config"
