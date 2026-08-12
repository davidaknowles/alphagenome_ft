#!/bin/bash
# Complete the matched HDA joint epoch-three LoRA plus LoCon result.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
lora_metrics=checkpoints/v0data/hda-joint_lora/last/metrics.json
locon_source=checkpoints/v0data/hda-joint_lora_locon/last

test "$(jq -er '.epoch' "$lora_metrics")" -eq 3
test "$(jq -er '.epoch' "$locon_source/metrics.json")" -eq 2
if [[ -d "$locon_source/optimizer_state" ]]; then
  echo "Expected the matched epoch-three boundary to restart AdamW state." >&2
  exit 2
fi

job=$(
  "$sbatch_bin" --parsable --nice="${NICE:-5}" --array=1 \
    --export="ALL,DATASET=hda-joint,RESUME_FROM=${locon_source},NUM_EPOCHS=3" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch
)
printf 'hda_joint_lora_locon_epoch3=%s\n' "$job"
