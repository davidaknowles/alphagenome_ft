#!/bin/bash
set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/joint-all-nonencode
"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_joint_dataset_config.py

smoke_job=$(
  sbatch --parsable \
    --export=ALL,SMOKE=1,RUN_SUFFIX=_provisional \
    scripts/v0data/slurm_joint_multidataset_adapters.sbatch
)
full_job=$(
  sbatch --parsable \
    --dependency="afterok:${smoke_job}" \
    --export=ALL,RUN_SUFFIX=_provisional \
    scripts/v0data/slurm_joint_multidataset_adapters.sbatch
)
printf 'smoke=%s\nfull=%s\n' "$smoke_job" "$full_job"
