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
    "$snapshot" "$learning_rate" "$dataset_config" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "source_epoch": int(sys.argv[3]),
    "source_checkpoint": sys.argv[4],
    "reset_optimizer": True,
    "learning_rate": float(sys.argv[5]),
    "dataset_config": sys.argv[6],
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
  exports="ALL,RUN_SUFFIX=${run_suffix},RESUME_FROM=${snapshot},RESET_OPTIMIZER=1,LEARNING_RATE=${learning_rate},NUM_EPOCHS=${num_epochs},DATASET_CONFIG=${dataset_config},TARGET_WORKERS=${target_workers},WINDOW_WORKERS=${window_workers}"
  local smoke full
  smoke=$(
    "$sbatch_bin" --parsable --array="$task" --time=00:30:00 \
      --export="${exports},SMOKE=1" scripts/v0data/slurm_joint_multidataset_adapters.sbatch
  )
  full=$(
    "$sbatch_bin" --parsable --array="$task" --dependency="afterok:${smoke}_${task}" \
      --export="$exports" scripts/v0data/slurm_joint_multidataset_adapters.sbatch
  )
  printf '%s smoke=%s full=%s\n' "$run_suffix" "$smoke" "$full"
}
