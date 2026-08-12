#!/bin/bash
# Submit a matched Zemke 2024 RNA-only correlation-objective screen.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data/objective-screens
sbatch_bin="${SBATCH_BIN:-sbatch}"
weight="${CORRELATION_LOSS_WEIGHT:-10}"
suffix="${weight//./p}"
targets="outputs/v0data/objective-screens/zemke2024-all_rna_corrw${suffix}.json"

"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_head_objective_config.py \
  --input outputs/v0data/zemke2024-all/targets.json \
  --output "$targets" \
  --head zemke2024_all_rna \
  --correlation-loss-weight "$weight"

exports="ALL,TARGETS_CONFIG=${targets},RUN_SUFFIX=_rna_corrw${suffix},NUM_EPOCHS=1"
smoke=$(
  "$sbatch_bin" --parsable --nice="${NICE:-20}" --time=00:30:00 --array=4-5%2 \
    --export="${exports},SMOKE=1" scripts/v0data/slurm_study_adapter_comparison.sbatch
)
full=$(
  "$sbatch_bin" --parsable --nice="${NICE:-20}" --array=4-5%2 \
    --dependency="afterok:${smoke}_*" --export="$exports" \
    scripts/v0data/slurm_study_adapter_comparison.sbatch
)
printf 'zemke2024 RNA correlation weight %s smoke=%s full=%s\n' "$weight" "$smoke" "$full"
