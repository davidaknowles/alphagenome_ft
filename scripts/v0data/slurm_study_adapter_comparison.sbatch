#!/bin/bash
#SBATCH --job-name=v0data_adapters
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/v0data/v0data_adapters-%A_%a.out
#SBATCH --error=logs/v0data/v0data_adapters-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data checkpoints/v0data

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/outputs/jax_compilation_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

DATASETS=(hda hda zemke2023-human zemke2023-human zemke2024-all zemke2024-all)
STRATEGIES=(lora lora+locon lora lora+locon lora lora+locon)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
STRATEGY="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"
TARGETS_CONFIG="${TARGETS_CONFIG:-outputs/v0data/${DATASET}/targets.json}"
RUN_NAME="${DATASET}_${STRATEGY//+/_}${RUN_SUFFIX:-}"
if [[ "${SMOKE:-0}" == "1" && "$RUN_NAME" != *_smoke ]]; then
  RUN_NAME="${RUN_NAME}_smoke"
fi
CANONICAL_CHROMS=chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX

if [[ ! -f "${TARGETS_CONFIG}" ]]; then
  echo "Missing target manifest: ${TARGETS_CONFIG}" >&2
  exit 2
fi

BEST_METRIC="valid/mean/differential_pearson_r"
TARGET_WORKER_COUNT="${TARGET_WORKERS:-24}"
WINDOW_WORKER_COUNT="${WINDOW_WORKERS:-8}"
if [[ "${DATASET}" == "hda" ]]; then
  BEST_METRIC="valid/differential_pearson_r"
elif [[ "${DATASET}" == "zemke2024-all" ]]; then
  TARGET_WORKER_COUNT="${TARGET_WORKERS:-24}"
  WINDOW_WORKER_COUNT="${WINDOW_WORKERS:-1}"
fi

EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(
    --num-epochs "${NUM_EPOCHS:-1}"
    --max-train-steps "${MAX_TRAIN_STEPS:-1}"
    --limit-train 8
    --limit-valid 8
    --limit-test 8
  )
else
  EXTRA_ARGS+=(--num-epochs "${NUM_EPOCHS:-100}")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  EXTRA_ARGS+=(--wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN_NAME}")
  [[ -n "${WANDB_ENTITY:-}" ]] && EXTRA_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  EXTRA_ARGS+=(--resume-from "${RESUME_FROM}")
fi
if [[ "${REPORT_HEAD_GRADIENT_NORMS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--report-head-gradient-norms)
fi
if [[ -n "${CORRELATION_LOSS_WEIGHT:-}" ]]; then
  EXTRA_ARGS+=(
    --double-centered-correlation-loss-weight "${CORRELATION_LOSS_WEIGHT}"
  )
fi

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --targets-config "${TARGETS_CONFIG}" \
  --fasta-path "${FASTA_PATH:-/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa}" \
  --include-chroms "${CANONICAL_CHROMS}" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --checkpoint-dir "checkpoints/v0data/${RUN_NAME}" \
  --pretrained-head-initialization "${PRETRAINED_HEAD_INITIALIZATION:-none}" \
  --head-id "${DATASET//-/_}" \
  --split-source chromosome \
  --window-size 131072 \
  --stride 131072 \
  --batch-size "${BATCH_SIZE:-8}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --backbone-lora \
  --adapter-strategy "${STRATEGY}" \
  --lora-rank "${LORA_RANK:-16}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --locon-rank "${LOCON_RANK:-4}" \
  --locon-alpha "${LOCON_ALPHA:-1}" \
  --base-param-dtype "${BASE_PARAM_DTYPE:-float32}" \
  --lora-param-dtype "${LORA_PARAM_DTYPE:-float32}" \
  --locon-param-dtype "${LOCON_PARAM_DTYPE:-float32}" \
  --activation-dtype "${ACTIVATION_DTYPE:-bfloat16}" \
  --backbone-compute-dtype "${BACKBONE_COMPUTE_DTYPE:-bfloat16}" \
  --base-compute-dtype "${BASE_COMPUTE_DTYPE:-bfloat16}" \
  --lora-compute-dtype "${LORA_COMPUTE_DTYPE:-bfloat16}" \
  --locon-compute-dtype "${LOCON_COMPUTE_DTYPE:-bfloat16}" \
  --best-metric "${BEST_METRIC}" \
  --best-metric-mode max \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-5}" \
  --eval-splits valid,test \
  --target-workers "${TARGET_WORKER_COUNT}" \
  --window-workers "${WINDOW_WORKER_COUNT}" \
  --prefetch-batches 2 \
  --progress-interval 100 \
  "${EXTRA_ARGS[@]}"
