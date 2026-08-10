#!/bin/bash
#SBATCH -J ag_allen_3sp
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 48:00:00
#SBATCH -o logs/allen_brain_multiome/%x-%j.out
#SBATCH -e logs/allen_brain_multiome/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/allen_brain_multiome_multispecies_v1}"
mkdir -p logs/allen_brain_multiome "${CHECKPOINT_DIR}"

export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async

EXTRA_ARGS=()
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max-train-steps "${MAX_TRAIN_STEPS}")
fi
if [[ -n "${LIMIT_TRAIN:-}" ]]; then
  EXTRA_ARGS+=(--limit-train "${LIMIT_TRAIN}")
fi
if [[ -n "${LIMIT_VALID:-}" ]]; then
  EXTRA_ARGS+=(--limit-valid "${LIMIT_VALID}")
fi
if [[ -n "${LIMIT_TEST:-}" ]]; then
  EXTRA_ARGS+=(--limit-test "${LIMIT_TEST}")
fi

PREPARED_DIR="${PREPARED_DIR:-outputs/allen_brain_multiome_multispecies_v1}"

python scripts/run_humanbraindev_finetune.py \
  --species-config "${PREPARED_DIR}/species.json" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --window-size "${WINDOW_SIZE:-131072}" \
  --stride "${STRIDE:-131072}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --drop-last \
  --num-epochs "${NUM_EPOCHS:-5}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --backbone-lora \
  --adapter-strategy lora+locon \
  --base-param-dtype bfloat16 \
  --lora-param-dtype bfloat16 \
  --locon-param-dtype bfloat16 \
  --best-metric valid_allen_rna/differential_pearson_r \
  --best-metric-mode max \
  --eval-splits valid,test \
  --target-workers "${TARGET_WORKERS:-8}" \
  --window-workers "${WINDOW_WORKERS:-4}" \
  --prefetch-batches 1 \
  --wandb-project "${WANDB_PROJECT:-alphagenome-allen-brain}" \
  --wandb-run-name "${WANDB_RUN_NAME:-allen_multiome_3species_exon_lora_locon}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  "${EXTRA_ARGS[@]}"
