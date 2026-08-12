#!/bin/bash
# Continue selected canonical Zemke species runs with early stopping.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
tasks="${TASKS:-2;3}"

task_metadata() {
  case "$1" in
    0) printf 'human lora\n' ;;
    1) printf 'human lora_locon\n' ;;
    2) printf 'macaque lora\n' ;;
    3) printf 'macaque lora_locon\n' ;;
    4) printf 'marmoset lora\n' ;;
    5) printf 'marmoset lora_locon\n' ;;
    6) printf 'mouse lora\n' ;;
    7) printf 'mouse lora_locon\n' ;;
    *) printf 'Unknown task %s.\n' "$1" >&2; return 2 ;;
  esac
}

IFS=';' read -r -a selected_tasks <<<"$tasks"
for task in "${selected_tasks[@]}"; do
  read -r species strategy < <(task_metadata "$task")
  source="checkpoints/v0data/zemke2023_${species}_${strategy}/last"
  test -f "$source/metrics.json"
  if [[ "${REQUIRE_OPTIMIZER_STATE:-1}" == "1" ]]; then
    test -d "$source/optimizer_state"
  fi
  sbatch_args=(--parsable --nice="${NICE:-40}" --array="$task")
  if [[ -n "${DEPENDENCY:-}" ]]; then
    sbatch_args+=(--dependency="$DEPENDENCY")
  fi
  job=$(
    "$sbatch_bin" "${sbatch_args[@]}" \
      --export="ALL,RESUME_FROM=${source},NUM_EPOCHS=${NUM_EPOCHS:-100}" \
      scripts/v0data/slurm_zemke2023_adapter_matrix.sbatch
  )
  printf '%s_%s=%s\n' "$species" "$strategy" "$job"
done
