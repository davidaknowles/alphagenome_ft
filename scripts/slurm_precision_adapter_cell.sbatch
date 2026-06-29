#!/bin/bash
#SBATCH -J ag-jaxtorch-cell
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/slurm

source ~/venv/jax/bin/activate
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async

: "${AG_BACKEND:?Set AG_BACKEND=jax or torch}"
: "${AG_PRECISION:?Set AG_PRECISION=default or nvfp8}"
: "${AG_ADAPTER_STRATEGY:?Set AG_ADAPTER_STRATEGY=lora or lora+locon}"
: "${AG_OUTPUT_ROOT:=/gpfs/commons/home/daknowles/projects/alphagenome_fp4/outputs/precision_adapter_compare}"

RUN_NAME="${AG_BACKEND}_${AG_PRECISION}_${AG_ADAPTER_STRATEGY//+/_}"
if [[ -n "${WANDB_PROJECT:-}" && -z "${WANDB_GROUP:-}" ]]; then
  WANDB_GROUP="$(basename "${AG_OUTPUT_ROOT}")"
fi
AUTO_BATCH_FLAG="--auto-batch-size"
if [[ "${AUTO_BATCH_SIZE:-true}" == "false" || "${AUTO_BATCH_SIZE:-true}" == "0" ]]; then
  AUTO_BATCH_FLAG="--no-auto-batch-size"
fi

python scripts/run_precision_adapter_cell.py \
  --backend "${AG_BACKEND}" \
  --precision "${AG_PRECISION}" \
  --adapter-strategy "${AG_ADAPTER_STRATEGY}" \
  --output-root "${AG_OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --batch-size "${BATCH_SIZE:-12}" \
  "${AUTO_BATCH_FLAG}" \
  --auto-batch-max-attempts "${AUTO_BATCH_MAX_ATTEMPTS:-6}" \
  --auto-batch-vram-fraction "${AUTO_BATCH_VRAM_FRACTION:-0.6666666667}" \
  --num-epochs "${NUM_EPOCHS:-50}" \
  --best-metric "${BEST_METRIC:-valid_differential_pearson_r}" \
  --best-metric-mode "${BEST_METRIC_MODE:-max}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-3}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA:-0.0}" \
  ${MAX_TRAIN_STEPS:+--max-train-steps "$MAX_TRAIN_STEPS"} \
  ${LIMIT_TRAIN:+--limit-train "$LIMIT_TRAIN"} \
  ${LIMIT_VALID:+--limit-valid "$LIMIT_VALID"} \
  ${LIMIT_TEST:+--limit-test "$LIMIT_TEST"} \
  ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"} \
  ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
  ${WANDB_GROUP:+--wandb-group "$WANDB_GROUP"} \
  ${WANDB_TAGS:+--wandb-tags "$WANDB_TAGS"} \
  ${WANDB_JOB_TYPE:+--wandb-job-type "$WANDB_JOB_TYPE"} \
  ${WANDB_MODE:+--wandb-mode "$WANDB_MODE"} \
  --wandb-run-name "${WANDB_RUN_NAME:-${RUN_NAME}}"
