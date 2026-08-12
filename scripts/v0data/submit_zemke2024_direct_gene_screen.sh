#!/bin/bash
# Submit matched Zemke 2024 direct-gene runs after the representation audit.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
audit="results/v0data_zemke2024_gene_track_agreement.json"
targets="outputs/v0data/zemke2024-gene-supervision/targets.json"
test -f "$audit"
test -f "$targets"
jq -e '
  .direct_gene_groups == 18 and
  (.masked_gene_groups | sort) ==
    (["Astro1_all", "Astro2_all", "Micro1_all", "Micro2_all"] | sort) and
  .raw_cpm_double_centered_r >= .minimum_r
' "$audit" >/dev/null

exports="ALL,DATASET=zemke2024-all,TARGETS_CONFIG=${targets},RUN_SUFFIX=_directgene_corrw10_screen,NUM_EPOCHS=1,BALANCE_GENE_WINDOWS=1"
smoke=$("$sbatch_bin" --parsable --nice="${NICE:-90}" --time=00:30:00 \
  --cpus-per-task=8 --array=0-1%2 --export="${exports},SMOKE=1" \
  scripts/v0data/slurm_joint_adapter_comparison.sbatch)
full=$("$sbatch_bin" --parsable --nice="${NICE:-90}" --time=12:00:00 \
  --array=0-1%2 --dependency="afterok:${smoke}_*" --export="$exports" \
  scripts/v0data/slurm_joint_adapter_comparison.sbatch)
printf 'Zemke 2024 direct-gene weight-10 smoke=%s full=%s\n' "$smoke" "$full"
