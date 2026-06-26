#!/bin/bash
#SBATCH -J ag_hbd_cmp
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
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
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

detect_gpu_mem_mb() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | awk '{print int($1)}'
}

auto_batch_size() {
  local kind="$1"
  local mem_mb="${GPU_MEM_MB:-$(detect_gpu_mem_mb)}"
  if [[ -z "${mem_mb}" || "${mem_mb}" -le 0 ]]; then
    if [[ "${kind}" == "heads" ]]; then
      echo 16
    else
      echo 8
    fi
    return
  fi

  if [[ "${mem_mb}" -ge 90000 ]]; then
    if [[ "${kind}" == "heads" ]]; then
      echo "${AUTO_BATCH_SIZE_HEADS_96G:-24}"
    else
      echo "${AUTO_BATCH_SIZE_LORA_96G:-12}"
    fi
  elif [[ "${mem_mb}" -ge 70000 ]]; then
    if [[ "${kind}" == "heads" ]]; then
      echo "${AUTO_BATCH_SIZE_HEADS_80G:-20}"
    else
      echo "${AUTO_BATCH_SIZE_LORA_80G:-10}"
    fi
  else
    if [[ "${kind}" == "heads" ]]; then
      echo "${AUTO_BATCH_SIZE_HEADS_DEFAULT:-16}"
    else
      echo "${AUTO_BATCH_SIZE_LORA_DEFAULT:-8}"
    fi
  fi
}

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}"
if [[ ! -d "${CHECKPOINT_PATH}" && ! -f "${HOME}/.kaggle/access_token" && -z "${KAGGLE_API_TOKEN:-}" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
  echo "No CHECKPOINT_PATH directory or Kaggle token found; cannot load base AlphaGenome weights." >&2
  echo "Set CHECKPOINT_PATH=/path/to/local/alphagenome/checkpoint or provide ~/.kaggle/access_token." >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    CONDITION="heads_default"
    BATCH_SIZE_VALUE="${BATCH_SIZE_HEADS:-$(auto_batch_size heads)}"
    METHOD_ARGS=()
    BASE_PARAM_DTYPE_VALUE="float32"
    LORA_PARAM_DTYPE_VALUE="float32"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="bfloat16"
    LORA_COMPUTE_DTYPE_VALUE="bfloat16"
    ;;
  1)
    CONDITION="lora_default"
    BATCH_SIZE_VALUE="${BATCH_SIZE_LORA:-$(auto_batch_size lora)}"
    METHOD_ARGS=(--backbone-lora)
    BASE_PARAM_DTYPE_VALUE="float32"
    LORA_PARAM_DTYPE_VALUE="float32"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="bfloat16"
    LORA_COMPUTE_DTYPE_VALUE="bfloat16"
    ;;
  2)
    CONDITION="heads_lora32_base16"
    BATCH_SIZE_VALUE="${BATCH_SIZE_HEADS:-$(auto_batch_size heads)}"
    METHOD_ARGS=()
    BASE_PARAM_DTYPE_VALUE="bf16"
    LORA_PARAM_DTYPE_VALUE="float32"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="bfloat16"
    LORA_COMPUTE_DTYPE_VALUE="bfloat16"
    ;;
  3)
    CONDITION="lora_lora32_base16"
    BATCH_SIZE_VALUE="${BATCH_SIZE_LORA:-$(auto_batch_size lora)}"
    METHOD_ARGS=(--backbone-lora)
    BASE_PARAM_DTYPE_VALUE="bf16"
    LORA_PARAM_DTYPE_VALUE="float32"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="bfloat16"
    LORA_COMPUTE_DTYPE_VALUE="bfloat16"
    ;;
  4)
    CONDITION="heads_lora16_base8"
    BATCH_SIZE_VALUE="${BATCH_SIZE_HEADS:-$(auto_batch_size heads)}"
    METHOD_ARGS=()
    BASE_PARAM_DTYPE_VALUE="fp16"
    LORA_PARAM_DTYPE_VALUE="fp16"
    ACTIVATION_DTYPE_VALUE="bfloat16"
    BASE_COMPUTE_DTYPE_VALUE="bfloat16"
    LORA_COMPUTE_DTYPE_VALUE="fp8"
    ;;
  5)
    CONDITION="lora_lora16_base8"
    BATCH_SIZE_VALUE="${BATCH_SIZE_LORA:-$(auto_batch_size lora)}"
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
echo "gpu_mem_mb=${GPU_MEM_MB:-$(detect_gpu_mem_mb)}"
echo "batch_size=${BATCH_SIZE:-${BATCH_SIZE_VALUE}}"

BIGWIG_DIR="${BIGWIG_DIR:-/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs}"
DATA_DIR="$(cd "${BIGWIG_DIR}/.." && pwd)"
TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-${DATA_DIR}/alphagenome_target_cache/humanbraindev_atac_w${WINDOW_SIZE:-131072}_${TARGET_CACHE_DTYPE:-float16}}"

echo "bigwig_dir=${BIGWIG_DIR}"
echo "target_cache_dir=${TARGET_CACHE_DIR}"

python scripts/run_humanbraindev_finetune.py \
  "${METHOD_ARGS[@]}" \
  --bigwig-dir "${BIGWIG_DIR}" \
  --fasta-path "${FASTA_PATH:-/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --checkpoint-dir "checkpoints/${RUN_DIR}" \
  --split-source "${SPLIT_SOURCE:-chromosome}" \
  --window-size "${WINDOW_SIZE:-131072}" \
  --stride "${STRIDE:-131072}" \
  --batch-size "${BATCH_SIZE:-${BATCH_SIZE_VALUE}}" \
  --num-epochs "${NUM_EPOCHS:-5}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-2}" \
  --num-devices "${NUM_DEVICES:-1}" \
  --progress-interval "${PROGRESS_INTERVAL:-50}" \
  --prefetch-batches "${PREFETCH_BATCHES:-2}" \
  --target-workers "${TARGET_WORKERS:-8}" \
  --target-cache-dir "${TARGET_CACHE_DIR}" \
  --target-cache-dtype "${TARGET_CACHE_DTYPE:-float16}" \
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
