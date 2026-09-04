#!/bin/bash

snapshot_checkpoint() {
  local source_checkpoint=$1
  local expected_epoch=$2
  local snapshot=$3
  local actual_epoch
  actual_epoch=$(jq -er '.epoch' "$source_checkpoint/metrics.json")
  if [[ "$actual_epoch" -ne "$expected_epoch" ]]; then
    printf 'Expected epoch %s at %s, found %s.\n' \
      "$expected_epoch" "$source_checkpoint" "$actual_epoch" >&2
    return 1
  fi
  if [[ ! -e "$snapshot" ]]; then
    cp -a "$source_checkpoint" "$snapshot"
  fi
  test "$(jq -er '.epoch' "$snapshot/metrics.json")" -eq "$expected_epoch"
}

prepare_history() {
  local source_metrics=$1
  local destination=$2
  local expected_epoch=$3
  "${HOME}/venv/jax/bin/python" - "$source_metrics" "$destination" "$expected_epoch" <<'PY'
import json
import sys
from pathlib import Path

source, destination, expected_epoch = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
records = [record for record in records if int(record["epoch"]) <= expected_epoch]
if not records or int(records[-1]["epoch"]) != expected_epoch:
    raise SystemExit(f"Metric history does not end at epoch {expected_epoch}.")
payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists() and destination.read_text() != payload:
    raise SystemExit(f"Refusing to replace mismatched metric history at {destination}.")
destination.write_text(payload)
PY
}

submit_continuation() {
  local task=$1
  local source_run=$2
  local source_epoch=$3
  local snapshot=$4
  local learning_rate=$5
  local run_suffix=$6
  local dataset_config=$7
  local strategy
  if [[ "$task" -eq 0 ]]; then strategy=lora; else strategy=lora_locon; fi
  local run_dir="checkpoints/v0data/joint_all_nonencode_${strategy}${run_suffix}"
  local smoke_dir="${run_dir}_smoke"

  prepare_history "$source_run/metrics.jsonl" "$run_dir/metrics.jsonl" "$source_epoch"
  prepare_history "$source_run/metrics.jsonl" "$smoke_dir/metrics.jsonl" "$source_epoch"
  "${HOME}/venv/jax/bin/python" - \
    "$run_dir/continuation.json" "$smoke_dir/continuation.json" "$source_epoch" \
    "$snapshot" "$learning_rate" "$dataset_config" \
    "${freeze_backbone_adapters:-0}" "${expand_backbone_adapters:-0}" \
    "${lora_rank:-16}" "${lora_alpha:-16}" "${locon_rank:-4}" \
    "${locon_alpha:-1}" "${locon_targets:-default}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "source_epoch": int(sys.argv[3]),
    "source_checkpoint": sys.argv[4],
    "reset_optimizer": True,
    "learning_rate": float(sys.argv[5]),
    "dataset_config": sys.argv[6],
    "freeze_backbone_adapters": bool(int(sys.argv[7])),
    "expand_backbone_adapters": bool(int(sys.argv[8])),
    "lora_rank": int(sys.argv[9]),
    "lora_alpha": float(sys.argv[10]),
    "locon_rank": int(sys.argv[11]),
    "locon_alpha": float(sys.argv[12]),
    "locon_targets": sys.argv[13],
}
serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
for value in sys.argv[1:3]:
    destination = Path(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != serialized:
        raise SystemExit(f"Refusing to replace mismatched provenance at {destination}.")
    destination.write_text(serialized)
PY

  local exports
  local cache_exports=""
  local gpu_gres="${GPU_GRES:-gpu:l40s:2}"
  if [[ -n "${TARGET_CACHE_DIR:-}" ]]; then
    cache_exports=",TARGET_CACHE_DIR=${TARGET_CACHE_DIR},TARGET_CACHE_SPLITS=${TARGET_CACHE_SPLITS:-valid;test},TARGET_CACHE_DTYPE=${TARGET_CACHE_DTYPE:-float16}"
  fi
  exports="ALL,RUN_SUFFIX=${run_suffix},RESUME_FROM=${snapshot},RESET_OPTIMIZER=1,LEARNING_RATE=${learning_rate},NUM_EPOCHS=${num_epochs},BALANCE_GENE_WINDOWS=${balance_gene_windows:-0},DATASET_CONFIG=${dataset_config},TARGET_WORKERS=${target_workers},WINDOW_WORKERS=${window_workers},SMOKE_LIMIT_TRAIN=${SMOKE_LIMIT_TRAIN:-8},SMOKE_LIMIT_VALID=${SMOKE_LIMIT_VALID:-8},SMOKE_LIMIT_TEST=${SMOKE_LIMIT_TEST:-8},FREEZE_BACKBONE_ADAPTERS=${freeze_backbone_adapters:-0},EXPAND_BACKBONE_ADAPTERS=${expand_backbone_adapters:-0},LORA_RANK=${lora_rank:-16},LORA_ALPHA=${lora_alpha:-16},LOCON_RANK=${locon_rank:-4},LOCON_ALPHA=${locon_alpha:-1},LOCON_TARGETS=${locon_targets:-default}${cache_exports}"
  local smoke full
  local smoke_args=(--parsable --array="$task" --time=00:30:00 --gres="$gpu_gres")
  if [[ -n "${initial_dependency:-}" ]]; then
    smoke_args+=(--dependency="$initial_dependency")
  fi
  smoke=$(
    "$sbatch_bin" "${smoke_args[@]}" \
      --export="${exports},SMOKE=1" scripts/v0data/slurm_joint_multidataset_adapters.sbatch
  )
  full=$(
    "$sbatch_bin" --parsable --array="$task" --gres="$gpu_gres" --dependency="afterok:${smoke}_${task}" \
      --export="$exports" scripts/v0data/slurm_joint_multidataset_adapters.sbatch
  )
  submitted_smoke_job="$smoke"
  submitted_full_job="$full"
  printf '%s smoke=%s full=%s\n' "$run_suffix" "$smoke" "$full"
}
