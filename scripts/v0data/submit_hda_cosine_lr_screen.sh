#!/bin/bash
# Screen a lower warmup-cosine learning rate on Mannens HDA with LoRA.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
common="DATASET=hda-joint,TARGETS_CONFIG=outputs/v0data/hda-joint/targets.json,RUN_SUFFIX=_cosine3e4_screen,LEARNING_RATE=3e-4,LEARNING_RATE_SCHEDULE=warmup_cosine,MINIMUM_LEARNING_RATE_RATIO=0.1,NUM_DEVICES=2,DEFER_TEST_EVALUATION=1"

smoke=$(
  "$sbatch_bin" --parsable --nice="${NICE:-50}" --gres=gpu:l40s:2 \
    --time=00:30:00 --array=0 \
    --export="ALL,${common},WARMUP_STEPS=0,SMOKE=1" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch
)
full=$(
  "$sbatch_bin" --parsable --nice="${NICE:-50}" --gres=gpu:l40s:2 \
    --array=0 --dependency="afterok:${smoke}_*" \
    --export="ALL,${common},WARMUP_STEPS=262,NUM_EPOCHS=8,EARLY_STOPPING_PATIENCE=5" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch
)
printf 'hda_cosine3e4_lora_smoke=%s full=%s\n' "$smoke" "$full"
