#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data outputs/v0data/joint-objective-variants

sbatch_bin="${SBATCH_BIN:-sbatch}"
num_epochs="${NUM_EPOCHS:-100}"
target_workers="${TARGET_WORKERS:-12}"
window_workers="${WINDOW_WORKERS:-4}"
freeze_backbone_adapters=1
source scripts/v0data/joint_continuation_lib.sh

source_run="${SOURCE_RUN:-checkpoints/v0data/joint_all_nonencode_lora_locon_lr3e5_epoch31_metric_tempered_reset}"
source_epoch="${SOURCE_EPOCH:-32}"
source_checkpoint="${SOURCE_CHECKPOINT:-${source_run}/best}"
snapshot="checkpoints/v0data/joint_all_nonencode_lora_locon_head_refit_seed_epoch${source_epoch}"
tempered_config="$(realpath outputs/v0data/joint-objective-variants/metric-tempered/datasets.json)"

snapshot_checkpoint "$source_checkpoint" "$source_epoch" "$snapshot"
test -f "$tempered_config"

submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" 1e-4 \
  _head_refit_lr1e4_epoch32_reset "$tempered_config"
submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" 3e-4 \
  _head_refit_lr3e4_epoch32_reset "$tempered_config"
submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" 1e-3 \
  _head_refit_lr1e3_epoch32_reset "$tempered_config"
