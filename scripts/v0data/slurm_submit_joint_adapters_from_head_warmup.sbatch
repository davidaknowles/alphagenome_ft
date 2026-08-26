#!/bin/bash
#SBATCH --job-name=v0_branch_warm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v0data/v0_branch_warm-%j.out
#SBATCH --error=logs/v0data/v0_branch_warm-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
source scripts/v0data/joint_continuation_lib.sh

sbatch_bin="${SBATCH_BIN:-sbatch}"
source_run="${SOURCE_RUN:?SOURCE_RUN is required}"
source_checkpoint="${source_run}/best"
source_epoch="$(jq -er '.epoch' "${source_checkpoint}/metrics.json")"
dataset_config="${DATASET_CONFIG:?DATASET_CONFIG is required}"
snapshot="checkpoints/v0data/joint_all_nonencode_head_warmup_seed_epoch${source_epoch}"
learning_rate=3e-4
num_epochs="${NUM_EPOCHS:-100}"
target_workers="${TARGET_WORKERS:-12}"
window_workers="${WINDOW_WORKERS:-4}"
expand_backbone_adapters=1
lora_rank=16
lora_alpha=16
locon_rank=4
locon_alpha=1
locon_targets=default

snapshot_checkpoint "$source_checkpoint" "$source_epoch" "$snapshot"
test -f "$dataset_config"

suffix="_headwarm${source_epoch}_adapters_lr3e4_reset"
submit_continuation 0 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  "$suffix" "$dataset_config"
submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  "$suffix" "$dataset_config"
