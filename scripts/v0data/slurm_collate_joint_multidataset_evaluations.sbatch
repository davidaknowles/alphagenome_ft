#!/bin/bash
#SBATCH --job-name=v0_joint_collate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/v0data/v0_joint_collate-%j.out
#SBATCH --error=logs/v0data/v0_joint_collate-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data results
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

dataset_config="${DATASET_CONFIG:-outputs/v0data/joint-all-nonencode/datasets.json}"
evaluation_root="${EVALUATION_ROOT:-evaluations/v0data/joint_all_nonencode}"
lora_run="${LORA_EVALUATION_RUN:?LORA_EVALUATION_RUN is required}"
locon_run="${LOCON_EVALUATION_RUN:?LOCON_EVALUATION_RUN is required}"
result_stem="${RESULT_STEM:?RESULT_STEM is required}"

"${HOME}/venv/jax/bin/python" scripts/v0data/collate_joint_multidataset_evaluations.py \
  --dataset-config "$dataset_config" \
  --evaluation-root "$evaluation_root" \
  --run "${lora_run}=LoRA" \
  --run "${locon_run}=LoRA+LoCon" \
  --json-output "results/${result_stem}.json" \
  --markdown-output "results/${result_stem}.md"
