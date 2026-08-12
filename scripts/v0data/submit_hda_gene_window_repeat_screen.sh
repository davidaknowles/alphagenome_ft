#!/bin/bash
# Compare one Mannens LoRA epoch with zero or one extra copy of gene-bearing windows.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"

for repeats in 0 1; do
  exports="ALL,DATASET=hda-joint,TARGETS_CONFIG=outputs/v0data/hda-joint/targets.json,RUN_SUFFIX=_generepeat${repeats}_screen,NUM_EPOCHS=1,BALANCE_GENE_WINDOWS=1,GENE_WINDOW_REPEATS=${repeats}"
  smoke=$("$sbatch_bin" --parsable --nice="${NICE:-10}" --time=00:30:00 --array=0 \
    --export="${exports},SMOKE=1" scripts/v0data/slurm_joint_adapter_comparison.sbatch)
  full=$("$sbatch_bin" --parsable --nice="${NICE:-10}" --time=12:00:00 --array=0 \
    --dependency="afterok:${smoke}_*" --export="$exports" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch)
  printf 'hda gene-window repeats %s smoke=%s full=%s\n' "$repeats" "$smoke" "$full"
done
