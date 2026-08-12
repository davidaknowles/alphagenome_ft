#!/bin/bash
# Submit a matched deterministic HDA RNA-only isolation screen.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data/objective-screens
sbatch_bin="${SBATCH_BIN:-sbatch}"
targets=outputs/v0data/objective-screens/hda-joint_rna_only.json

"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_single_head_config.py \
  --input outputs/v0data/hda-joint/targets.json \
  --output "$targets" \
  --head hda_rna

exports="ALL,DATASET=hda-joint,TARGETS_CONFIG=${targets},RUN_SUFFIX=_rna_only_neural_accessibility_bootstrap_screen,NUM_EPOCHS=1,PRETRAINED_HEAD_INITIALIZATION=neural_accessibility_bootstrap"
smoke=$(
  "$sbatch_bin" --parsable --nice="${NICE:-80}" --time=00:30:00 --array=0-1%2 \
    --export="${exports},SMOKE=1" scripts/v0data/slurm_joint_adapter_comparison.sbatch
)
full=$(
  "$sbatch_bin" --parsable --nice="${NICE:-80}" --array=0-1%2 \
    --dependency="afterok:${smoke}_*" --export="$exports" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch
)
printf 'hda RNA-only isolation smoke=%s full=%s\n' "$smoke" "$full"
