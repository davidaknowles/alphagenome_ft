#!/bin/bash
set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data

run_suffix="${RUN_SUFFIX:-_provisional}"
lora_job="${LORA_JOB:-}"
locon_job="${LOCON_JOB:-}"
checkpoint_root="${CHECKPOINT_ROOT:-checkpoints/v0data}"

if { [[ -n "$lora_job" ]] && [[ -z "$locon_job" ]]; } || \
  { [[ -z "$lora_job" ]] && [[ -n "$locon_job" ]]; }; then
  printf 'Set both LORA_JOB and LOCON_JOB, or neither.\n' >&2
  exit 1
fi

for strategy in lora lora_locon; do
  checkpoint="${checkpoint_root}/joint_all_nonencode_${strategy}${run_suffix}/best/metrics.json"
  if [[ ! -f "$checkpoint" ]]; then
    printf 'Missing evaluation checkpoint metadata: %s\n' "$checkpoint" >&2
    exit 1
  fi
done

sbatch_args=(
  --parsable
  --export="ALL,RUN_SUFFIX=${run_suffix},CHECKPOINT_ROOT=${checkpoint_root}"
)
if [[ -n "$lora_job" ]]; then
  sbatch_args+=(--dependency="afterok:${lora_job}:${locon_job}")
fi
job=$(sbatch "${sbatch_args[@]}" scripts/v0data/slurm_joint_multidataset_evaluate.sbatch)
printf 'joint native-source evaluations=%s\n' "$job"
