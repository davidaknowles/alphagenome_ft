#!/bin/bash
# Continue selected corrected joint Johansen runs with preserved optimizer state.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
tasks="${TASKS:-0;1}"
species_config="${SPECIES_CONFIG:-outputs/v0data/johansen-rna-corrected/geneonly-corrw1/species.json}"
test -f "$species_config"

task_strategy() {
  case "$1" in
    0) printf 'lora\n' ;;
    1) printf 'lora_locon\n' ;;
    *) printf 'Unknown task %s.\n' "$1" >&2; return 2 ;;
  esac
}

IFS=';' read -r -a selected_tasks <<<"$tasks"
for task in "${selected_tasks[@]}"; do
  strategy="$(task_strategy "$task")"
  source="checkpoints/v0data/johansen_joint_${strategy}_rawcount_geneonly_corrw1/last"
  test -f "$source/metrics.json"
  test "$(jq -er '.epoch' "$source/metrics.json")" -ge "${MIN_SOURCE_EPOCH:-1}"
  if [[ "${REQUIRE_OPTIMIZER_STATE:-1}" == "1" ]]; then
    test -d "$source/optimizer_state"
  fi
  sbatch_args=(--parsable --nice="${NICE:-40}" --array="$task")
  if [[ -n "${DEPENDENCY:-}" ]]; then
    sbatch_args+=(--dependency="$DEPENDENCY")
  fi
  job=$(
    "$sbatch_bin" "${sbatch_args[@]}" \
      --export="ALL,SPECIES_CONFIG=${species_config},RESUME_FROM=${source},RUN_SUFFIX=_rawcount_geneonly_corrw1,NUM_EPOCHS=${NUM_EPOCHS:-100}" \
      scripts/v0data/slurm_johansen_joint_adapters.sbatch
  )
  printf '%s=%s\n' "$strategy" "$job"
done
