#!/bin/bash
#SBATCH --job-name=v0_joint_eval_submit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v0data/v0_joint_eval_submit-%j.out
#SBATCH --error=logs/v0data/v0_joint_eval_submit-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

dataset_config="${DATASET_CONFIG:?DATASET_CONFIG is required}"
lora_run="${LORA_RUN:?LORA_RUN is required}"
locon_run="${LOCON_RUN:?LOCON_RUN is required}"
evaluation_tag="${EVALUATION_TAG:?EVALUATION_TAG is required}"
gpu_gres="${GPU_GRES:-gpu:l40s:2}"
locon_targets="${LOCON_TARGETS:-default}"
target_workers="${TARGET_WORKERS:-24}"
window_workers="${WINDOW_WORKERS:-8}"

for checkpoint in "${lora_run}/best/metrics.json" "${locon_run}/best/metrics.json"; do
  test -f "$checkpoint"
done

lora_tag="${evaluation_tag}_lora"
locon_tag="${evaluation_tag}_lora_locon"
lora_submission=$( \
  STRATEGY=lora \
  SOURCE_CHECKPOINT="${lora_run}/best" \
  EVALUATION_TAG="$lora_tag" \
  DATASET_CONFIG="$dataset_config" \
  GPU_GRES="$gpu_gres" \
  TARGET_WORKERS="$target_workers" \
  WINDOW_WORKERS="$window_workers" \
  bash scripts/v0data/submit_joint_checkpoint_evaluations.sh
)
locon_submission=$( \
  STRATEGY=lora_locon \
  SOURCE_CHECKPOINT="${locon_run}/best" \
  EVALUATION_TAG="$locon_tag" \
  DATASET_CONFIG="$dataset_config" \
  GPU_GRES="$gpu_gres" \
  LOCON_TARGETS="$locon_targets" \
  TARGET_WORKERS="$target_workers" \
  WINDOW_WORKERS="$window_workers" \
  bash scripts/v0data/submit_joint_checkpoint_evaluations.sh
)
lora_evaluations="${lora_submission##*=}"
locon_evaluations="${locon_submission##*=}"

collation=$( \
  sbatch --parsable \
    --dependency="afterok:${lora_evaluations}:${locon_evaluations}" \
    --export="ALL,DATASET_CONFIG=${dataset_config},EVALUATION_TAG=${evaluation_tag},LORA_TAG=${lora_tag},LOCON_TAG=${locon_tag}" \
    scripts/v0data/slurm_collate_joint_selected_evaluations.sbatch
)
printf '%s lora-evaluations=%s locon-evaluations=%s collation=%s\n' \
  "$evaluation_tag" "$lora_evaluations" "$locon_evaluations" "$collation"
