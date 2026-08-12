#!/bin/bash
# Submit a matched human Zemke 2023 direct-gene RNA screen after target audit.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
audit="results/v0data_zemke2023_gene_target_agreement.json"
targets="outputs/v0data/zemke2023-gene-supervision/human/targets.json"
test -f "$audit"
test -f "$targets"
jq -e '
  (.species | length) == 4 and
  ([.species[].species] | sort) == (["human", "macaque", "marmoset", "mouse"] | sort) and
  all(.species[]; .raw_cpm_double_centered_r >= $minimum)
' --argjson minimum "$(jq '.minimum_r' "$audit")" "$audit" >/dev/null

weight="${CORRELATION_LOSS_WEIGHT:-10}"
suffix="${weight//./p}"
exports="ALL,DATASET=zemke2023-human,TARGETS_CONFIG=${targets},CORRELATION_LOSS_WEIGHT=${weight},RUN_SUFFIX=_directgene_corrw${suffix}_screen,NUM_EPOCHS=1"
smoke=$("$sbatch_bin" --parsable --nice="${NICE:-90}" --time=00:30:00 \
  --cpus-per-task=8 --array=0-1%2 --export="${exports},SMOKE=1" \
  scripts/v0data/slurm_joint_adapter_comparison.sbatch)
full=$("$sbatch_bin" --parsable --nice="${NICE:-90}" --time=12:00:00 \
  --array=0-1%2 --dependency="afterok:${smoke}_*" --export="$exports" \
  scripts/v0data/slurm_joint_adapter_comparison.sbatch)
printf 'Zemke 2023 human direct-gene corr weight %s smoke=%s full=%s\n' \
  "$weight" "$smoke" "$full"
