#!/bin/bash
# Evaluate selected joint Zemke checkpoints on each native species dataset.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"

submit_evaluation() {
  local task=$1
  local strategy=$2
  local species=$3
  local source="checkpoints/v0data/zemke2023_joint_${strategy}/best"
  local epoch
  test -f "$source/metrics.json"
  epoch=$(jq -er '.epoch | select(type == "number" and . >= 1)' "$source/metrics.json")
  "$sbatch_bin" --parsable --nice="${NICE:-30}" --time="${TIME_LIMIT:-01:00:00}" \
    --array="$task" \
    --export="ALL,EVALUATE_ONLY=1,EVALUATE_SPECIES=${species},RESUME_FROM=${source},RUN_SUFFIX=_joint_epoch${epoch}_eval" \
    scripts/v0data/slurm_zemke2023_joint_adapters.sbatch
}

for species in human macaque marmoset mouse; do
  printf 'lora_%s=%s\n' "$species" "$(submit_evaluation 0 lora "$species")"
  printf 'lora_locon_%s=%s\n' "$species" "$(submit_evaluation 1 lora_locon "$species")"
done
