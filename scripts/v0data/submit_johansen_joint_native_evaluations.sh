#!/bin/bash
# Evaluate corrected joint Johansen checkpoints on each native species dataset.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
source_job="${SOURCE_JOB:-19779728}"
species_config="${SPECIES_CONFIG:-outputs/v0data/johansen-rna-corrected/geneonly-corrw1/species.json}"

submit_evaluation() {
  local task=$1
  local strategy=$2
  local species=$3
  local source="checkpoints/v0data/johansen_joint_${strategy}_rawcount_geneonly_corrw1/best"
  "$sbatch_bin" --parsable --nice="${NICE:-30}" --time="${TIME_LIMIT:-01:00:00}" \
    --array="$task" --dependency="afterok:${source_job}_${task}" \
    --export="ALL,SPECIES_CONFIG=${species_config},EVALUATE_ONLY=1,EVALUATE_SPECIES=${species},RESUME_FROM=${source},RUN_SUFFIX=_${species}_eval" \
    scripts/v0data/slurm_johansen_joint_adapters.sbatch
}

for species in human macaque marmoset; do
  printf 'lora_%s=%s\n' "$species" "$(submit_evaluation 0 lora "$species")"
  printf 'lora_locon_%s=%s\n' "$species" "$(submit_evaluation 1 lora_locon "$species")"
done
