#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data outputs/v0data/joint-objective-variants

sbatch_bin="${SBATCH_BIN:-sbatch}"
num_epochs="${NUM_EPOCHS:-100}"
target_workers="${TARGET_WORKERS:-12}"
window_workers="${WINDOW_WORKERS:-4}"
source scripts/v0data/joint_continuation_lib.sh

lora_source="checkpoints/v0data/joint_all_nonencode_lora_provisional"
lora_epoch=9
lora_snapshot="checkpoints/v0data/joint_all_nonencode_lora_lr3e4_seed_epoch${lora_epoch}"
locon_source="checkpoints/v0data/joint_all_nonencode_lora_locon_lr3e4_reset"
locon_epoch=17
locon_snapshot="checkpoints/v0data/joint_all_nonencode_lora_locon_lr1e4_seed_epoch${locon_epoch}"
canonical_config="$(realpath outputs/v0data/joint-all-nonencode/datasets.json)"
rna_weight_root="outputs/v0data/joint-objective-variants/rnaw2"

"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_joint_rna_objective_config.py \
  --input "$canonical_config" \
  --output-dir "$rna_weight_root" \
  --loss-weight 2
rna_weight_config="$(realpath "$rna_weight_root/datasets.json")"

snapshot_checkpoint "$lora_source/best" "$lora_epoch" "$lora_snapshot"
snapshot_checkpoint "$locon_source/best" "$locon_epoch" "$locon_snapshot"

test -f "$canonical_config"
test -f "$rna_weight_config"

submit_continuation 0 "$lora_source" "$lora_epoch" "$lora_snapshot" 3e-4 \
  _lr3e4_reset "$canonical_config"
submit_continuation 1 "$locon_source" "$locon_epoch" "$locon_snapshot" 1e-4 \
  _lr1e4_reset "$canonical_config"
submit_continuation 1 "$locon_source" "$locon_epoch" "$locon_snapshot" 1e-4 \
  _lr1e4_rnaw2_reset "$rna_weight_config"
