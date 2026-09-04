#!/bin/bash
#SBATCH --job-name=v0_joint_eval_collate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v0data/v0_joint_eval_collate-%j.out
#SBATCH --error=logs/v0data/v0_joint_eval_collate-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

dataset_config="${DATASET_CONFIG:?DATASET_CONFIG is required}"
evaluation_tag="${EVALUATION_TAG:?EVALUATION_TAG is required}"
lora_tag="${LORA_TAG:?LORA_TAG is required}"
locon_tag="${LOCON_TAG:?LOCON_TAG is required}"

"${HOME}/venv/jax/bin/python" scripts/v0data/collate_joint_multidataset_evaluations.py \
  --dataset-config "$dataset_config" \
  --evaluation-root evaluations/v0data/joint_all_nonencode \
  --run "${lora_tag}=lora" \
  --run "${locon_tag}=lora+locon" \
  --json-output "results/v0data/${evaluation_tag}_native.json" \
  --markdown-output "results/v0data/${evaluation_tag}_native.md"
