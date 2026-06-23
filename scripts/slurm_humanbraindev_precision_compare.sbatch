#!/bin/bash
#SBATCH -J ag_hbd_cmp
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH --array=0-5
#SBATCH -o logs/humanbraindev_precision_compare/%x-%A_%a.out
#SBATCH -e logs/humanbraindev_precision_compare/%x-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/humanbraindev_precision_compare checkpoints/humanbraindev_precision_compare

NVIDIA_SITE="${HOME}/venv/jax/lib/python3.12/site-packages/nvidia"
if [[ -d "${NVIDIA_SITE}" ]]; then
  NVIDIA_LIBS="$(find "${NVIDIA_SITE}" -maxdepth 2 -type d -name lib | paste -sd: -)"
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}:/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -f "${HOME}/.kaggle/access_token" ]]; then
  export KAGGLE_API_TOKEN="$(< "${HOME}/.kaggle/access_token")"
fi

source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-daknowles}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}"
if [[ ! -d "${CHECKPOINT_PATH}" && ! -f "${HOME}/.kaggle/access_token" && -z "${KAGGLE_API_TOKEN:-}" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
  echo "No CHECKPOINT_PATH directory or Kaggle token found; cannot load base AlphaGenome weights." >&2
  echo "Set CHECKPOINT_PATH=/path/to/local/alphagenome/checkpoint or provide ~/.kaggle/access_token." >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    CONDITION="heads_default"
    METHOD_ARGS=()
    BASE_PARAM_DTYPE_VALUE="float32"
    LORA_PARAM_DTYPE_VALUE="float32"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="bfloat16"
    LORA_COMPUTE_DTYPE_VALUE="bfloat16"
    ;;
  1)
    CONDITION="lora_default"
    METHOD_ARGS=(--backbone-lora)
    BASE_PARAM_DTYPE_VALUE="float32"
    LORA_PARAM_DTYPE_VALUE="float32"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="bfloat16"
    LORA_COMPUTE_DTYPE_VALUE="bfloat16"
    ;;
  2)
    CONDITION="heads_lora32_base16"
    METHOD_ARGS=()
    BASE_PARAM_DTYPE_VALUE="bf16"
    LORA_PARAM_DTYPE_VALUE="float32"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="bfloat16"
    LORA_COMPUTE_DTYPE_VALUE="bfloat16"
    ;;
  3)
    CONDITION="lora_lora32_base16"
    METHOD_ARGS=(--backbone-lora)
    BASE_PARAM_DTYPE_VALUE="bf16"
    LORA_PARAM_DTYPE_VALUE="float32"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="bfloat16"
    LORA_COMPUTE_DTYPE_VALUE="bfloat16"
    ;;
  4)
    CONDITION="heads_lora16_base8"
    METHOD_ARGS=()
    BASE_PARAM_DTYPE_VALUE="fp16"
    LORA_PARAM_DTYPE_VALUE="fp16"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="bfloat16"
    LORA_COMPUTE_DTYPE_VALUE="fp8"
    ;;
  5)
    CONDITION="lora_lora16_base8"
    METHOD_ARGS=(--backbone-lora --fp8-lora)
    BASE_PARAM_DTYPE_VALUE="fp8"
    LORA_PARAM_DTYPE_VALUE="fp16"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="fp8"
    LORA_COMPUTE_DTYPE_VALUE="fp8"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

RUN_NAME="${CONDITION}${RUN_SUFFIX:-}"
RUN_DIR="humanbraindev_precision_compare/${RUN_NAME}"
mkdir -p "checkpoints/${RUN_DIR}"

echo "condition=${CONDITION}"
echo "run_name=${RUN_NAME}"
echo "checkpoint_path=${CHECKPOINT_PATH}"
echo "base_param_dtype=${BASE_PARAM_DTYPE_VALUE}"
echo "lora_param_dtype=${LORA_PARAM_DTYPE_VALUE}"
echo "activation_dtype=${ACTIVATION_DTYPE_VALUE}"
echo "base_compute_dtype=${BASE_COMPUTE_DTYPE_VALUE}"
echo "lora_compute_dtype=${LORA_COMPUTE_DTYPE_VALUE}"

python scripts/run_humanbraindev_finetune.py \
  "${METHOD_ARGS[@]}" \
  --bigwig-dir "${BIGWIG_DIR:-/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs}" \
  --fasta-path "${FASTA_PATH:-/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --checkpoint-dir "checkpoints/${RUN_DIR}" \
  --split-source "${SPLIT_SOURCE:-chromosome}" \
  --window-size "${WINDOW_SIZE:-131072}" \
  --stride "${STRIDE:-131072}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-epochs "${NUM_EPOCHS:-5}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-2}" \
  --num-devices "${NUM_DEVICES:-1}" \
  --base-param-dtype "${BASE_PARAM_DTYPE_VALUE}" \
  --lora-param-dtype "${LORA_PARAM_DTYPE_VALUE}" \
  --activation-dtype "${ACTIVATION_DTYPE_VALUE}" \
  --base-compute-dtype "${BASE_COMPUTE_DTYPE_VALUE}" \
  --lora-compute-dtype "${LORA_COMPUTE_DTYPE_VALUE}" \
  --wandb-run-name "${WANDB_RUN_NAME:-${RUN_NAME}}" \
  ${MAX_TRAIN_STEPS:+--max-train-steps "$MAX_TRAIN_STEPS"} \
  ${LIMIT_TRAIN:+--limit-train "$LIMIT_TRAIN"} \
  ${LIMIT_VALID:+--limit-valid "$LIMIT_VALID"} \
  ${LIMIT_TEST:+--limit-test "$LIMIT_TEST"} \
  ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"} \
  ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"}
