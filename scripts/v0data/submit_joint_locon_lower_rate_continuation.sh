#!/bin/bash
set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data checkpoints/v0data

source_run="${SOURCE_RUN:-checkpoints/v0data/joint_all_nonencode_lora_locon_provisional}"
source_checkpoint="${SOURCE_CHECKPOINT:-${source_run}/best}"
source_epoch="${SOURCE_EPOCH:-6}"
run_suffix="${RUN_SUFFIX:-_lr3e4_reset}"
run_dir="checkpoints/v0data/joint_all_nonencode_lora_locon${run_suffix}"
snapshot="checkpoints/v0data/joint_all_nonencode_lora_locon${run_suffix}_seed_epoch${source_epoch}"

"${HOME}/venv/jax/bin/python" - "$source_checkpoint" "$source_epoch" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
expected_epoch = int(sys.argv[2])
record = json.loads((checkpoint / "metrics.json").read_text())
if int(record["epoch"]) != expected_epoch:
    raise SystemExit(
        f"Expected source epoch {expected_epoch}, found {record['epoch']} in {checkpoint}."
    )
PY

if [[ ! -e "$snapshot" ]]; then
  cp -a "$source_checkpoint" "$snapshot"
fi
mkdir -p "$run_dir"
"${HOME}/venv/jax/bin/python" - "$source_run/metrics.jsonl" "$run_dir/metrics.jsonl" "$source_epoch" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
expected_epoch = int(sys.argv[3])
records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
records = [record for record in records if int(record["epoch"]) <= expected_epoch]
if not records or int(records[-1]["epoch"]) != expected_epoch:
    raise SystemExit(f"Metric history does not end at epoch {expected_epoch}.")
payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
if destination.exists() and destination.read_text() != payload:
    raise SystemExit(f"Refusing to replace mismatched metric history at {destination}.")
destination.write_text(payload)
PY

job=$(
  sbatch --parsable --array=1 \
    --export="ALL,RUN_SUFFIX=${run_suffix},RESUME_FROM=${snapshot},RESET_OPTIMIZER=1,LEARNING_RATE=${LEARNING_RATE:-3e-4}" \
    scripts/v0data/slurm_joint_multidataset_adapters.sbatch
)
printf 'joint LoRA plus LoCon lower-rate continuation=%s\n' "$job"
