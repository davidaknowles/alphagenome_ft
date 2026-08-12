#!/bin/bash
set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data

run_suffix="${RUN_SUFFIX:-}"
lora_job="${LORA_JOB:?Set LORA_JOB to the completed or active joint LoRA job ID}"
locon_job="${LOCON_JOB:?Set LOCON_JOB to the completed or active joint LoRA plus LoCon job ID}"

job=$(
  sbatch --parsable \
    --dependency="afterok:${lora_job}:${locon_job}" \
    --export="ALL,RUN_SUFFIX=${run_suffix}" \
    scripts/v0data/slurm_joint_multidataset_evaluate.sbatch
)
printf 'joint native-source evaluations=%s\n' "$job"
