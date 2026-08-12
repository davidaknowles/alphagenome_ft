#!/bin/bash
# Evaluate selected joint Zemke checkpoints on each native species dataset.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"

submit_strategy() {
  local strategy=$1
  local tasks=$2
  local source="checkpoints/v0data/zemke2023_joint_${strategy}/best"
  local epoch
  test -f "$source/metrics.json"
  epoch=$(jq -er '.epoch | select(type == "number" and . >= 1)' "$source/metrics.json")
  "$sbatch_bin" --parsable --nice="${NICE:-30}" --time="${TIME_LIMIT:-01:00:00}" \
    --array="$tasks" \
    --export="ALL,EVALUATE_ONLY=1,RESUME_FROM=${source},RUN_SUFFIX=_joint_epoch${epoch}_eval" \
    scripts/v0data/slurm_zemke2023_adapter_matrix.sbatch
}

printf 'lora=%s\n' "$(submit_strategy lora 0,2,4,6)"
printf 'lora_locon=%s\n' "$(submit_strategy lora_locon 1,3,5,7)"
