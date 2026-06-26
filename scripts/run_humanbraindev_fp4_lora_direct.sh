#!/bin/bash
set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/humanbraindev_fp4_lora checkpoints/humanbraindev_fp4_lora

NVIDIA_SITE="${HOME}/venv/jax/lib/python3.12/site-packages/nvidia"
NVIDIA_LIBS="$(find "${NVIDIA_SITE}" -maxdepth 2 -type d -name lib | paste -sd: -)"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-daknowles}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG_PATH="${LOG_PATH:-logs/humanbraindev_fp4_lora/nvfp4_lora_rank32_direct.log}"

exec > >(tee -a "${LOG_PATH}") 2>&1

echo "start_time=$(date -Is)"
echo "hostname=$(hostname)"
echo "log_path=${LOG_PATH}"
nvidia-smi

exec "${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --backbone-lora \
  --fp4-lora \
  --lora-rank "${LORA_RANK:-32}" \
  --lora-alpha "${LORA_ALPHA:-32.0}" \
  --base-param-dtype "${BASE_PARAM_DTYPE:-bfloat16}" \
  --lora-param-dtype "${LORA_PARAM_DTYPE:-float32}" \
  --activation-dtype "${ACTIVATION_DTYPE:-bfloat16}" \
  --base-compute-dtype "${BASE_COMPUTE_DTYPE:-bfloat16}" \
  --lora-compute-dtype "${LORA_COMPUTE_DTYPE:-fp4}" \
  --lora-targets "${LORA_TARGETS:-default}" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --checkpoint-dir "${CHECKPOINT_DIR:-checkpoints/humanbraindev_fp4_lora}" \
  --split-source "${SPLIT_SOURCE:-chromosome}" \
  --window-size "${WINDOW_SIZE:-131072}" \
  --stride "${STRIDE:-131072}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-epochs "${NUM_EPOCHS:-5}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-2}" \
  --num-devices "${NUM_DEVICES:-1}" \
  ${MAX_TRAIN_STEPS:+--max-train-steps "$MAX_TRAIN_STEPS"} \
  ${LIMIT_TRAIN:+--limit-train "$LIMIT_TRAIN"} \
  ${LIMIT_VALID:+--limit-valid "$LIMIT_VALID"} \
  ${LIMIT_TEST:+--limit-test "$LIMIT_TEST"}
