#!/bin/bash
# Screen published-track-integrated direct-gene supervision on Zemke human.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
source_job="${SOURCE_JOB:-19817129}"
targets="outputs/v0data/zemke2023-published-gene-supervision/human/targets.json"
suffix=_published_gene_corrw10

smoke=$("$sbatch_bin" --parsable --nice="${NICE:-20}" --time=00:30:00 \
  --array=0-1%2 --dependency="afterok:${source_job}_0" \
  --export="ALL,TARGETS_CONFIG=${targets},RUN_SUFFIX=${suffix},NUM_EPOCHS=1,SMOKE=1" \
  scripts/v0data/slurm_zemke2023_adapter_matrix.sbatch)

full=$("$sbatch_bin" --parsable --nice="${NICE:-20}" --array=0-1%2 \
  --dependency="afterok:${smoke}_*" \
  --export="ALL,TARGETS_CONFIG=${targets},RUN_SUFFIX=${suffix},NUM_EPOCHS=1" \
  scripts/v0data/slurm_zemke2023_adapter_matrix.sbatch)

for task in 0 1; do
  strategy=lora
  [[ "$task" == "1" ]] && strategy=lora_locon
  source="checkpoints/v0data/zemke2023_human_${strategy}${suffix}/best"
  evaluation=$("$sbatch_bin" --parsable --nice="${NICE:-20}" --time=01:00:00 \
    --array="$task" --dependency="afterok:${full}_${task}" \
    --export="ALL,TARGETS_CONFIG=outputs/v0data/zemke2023-species/human/targets.json,RESUME_FROM=${source},RUN_SUFFIX=${suffix}_coordinate_eval,EVALUATE_ONLY=1" \
    scripts/v0data/slurm_zemke2023_adapter_matrix.sbatch)
  printf 'coordinate_%s=%s\n' "$strategy" "$evaluation"
done

printf 'smoke=%s\nfull=%s\n' "$smoke" "$full"
