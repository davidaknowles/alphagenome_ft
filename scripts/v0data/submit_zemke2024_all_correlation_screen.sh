#!/bin/bash
# Submit a matched Zemke 2024 all-head correlation-objective screen.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
weight="${CORRELATION_LOSS_WEIGHT:-10}"
suffix="${weight//./p}"
exports="ALL,CORRELATION_LOSS_WEIGHT=${weight},RUN_SUFFIX=_all_corrw${suffix},NUM_EPOCHS=1"

smoke=$(
  "$sbatch_bin" --parsable --nice="${NICE:-30}" --time=00:30:00 --array=4-5%2 \
    --export="${exports},SMOKE=1" scripts/v0data/slurm_study_adapter_comparison.sbatch
)
full=$(
  "$sbatch_bin" --parsable --nice="${NICE:-30}" --array=4-5%2 \
    --dependency="afterok:${smoke}_*" --export="$exports" \
    scripts/v0data/slurm_study_adapter_comparison.sbatch
)
printf 'zemke2024 all-head correlation weight %s smoke=%s full=%s\n' \
  "$weight" "$smoke" "$full"
