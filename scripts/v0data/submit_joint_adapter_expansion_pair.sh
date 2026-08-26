#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data outputs/v0data/joint-objective-variants

sbatch_bin="${SBATCH_BIN:-sbatch}"
num_epochs="${NUM_EPOCHS:-100}"
target_workers="${TARGET_WORKERS:-12}"
window_workers="${WINDOW_WORKERS:-4}"
expand_backbone_adapters=1
source scripts/v0data/joint_continuation_lib.sh

source_run="${SOURCE_RUN:-checkpoints/v0data/joint_all_nonencode_lora_locon_lr3e5_epoch31_metric_tempered_reset}"
source_epoch="${SOURCE_EPOCH:-32}"
source_checkpoint="${SOURCE_CHECKPOINT:-${source_run}/best}"
snapshot="checkpoints/v0data/joint_all_nonencode_lora_locon_expansion_seed_epoch${source_epoch}"
learning_rate="${LEARNING_RATE:-1e-4}"
dataset_config="$(realpath outputs/v0data/joint-objective-variants/metric-tempered/datasets.json)"
locon_targets="downres_block_2;downres_block_3;downres_block_4;downres_block_5"

snapshot_checkpoint "$source_checkpoint" "$source_epoch" "$snapshot"
test -f "$dataset_config"

lora_rank=16
lora_alpha=16
locon_rank=4
locon_alpha=1
submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  _expanded_locon_blocks2to5_epoch32_reset "$dataset_config"

lora_rank=32
lora_alpha=32
locon_rank=8
locon_alpha=2
submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  _expanded_rank32_locon8_blocks2to5_epoch32_reset "$dataset_config"
