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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

DATASETS=(hda hda zemke2023-human zemke2023-human zemke2024-all zemke2024-all)
STRATEGIES=(lora lora+locon lora lora+locon lora lora+locon)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
STRATEGY="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"
TARGETS_CONFIG="outputs/v0data/${DATASET}/targets.json"
RUN_NAME="${DATASET}_${STRATEGY//+/_}"

if [[ ! -f "${TARGETS_CONFIG}" ]]; then
  echo "Missing target manifest: ${TARGETS_CONFIG}" >&2
  exit 2
fi

BEST_METRIC="valid/mean/differential_pearson_r"
if [[ "${DATASET}" == "hda" ]]; then
  BEST_METRIC="valid/differential_pearson_r"
fi

EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--num-epochs 1 --max-train-steps 1 --limit-train 8 --limit-valid 8 --limit-test 8)
else
  EXTRA_ARGS+=(--num-epochs "${NUM_EPOCHS:-100}")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  EXTRA_ARGS+=(--wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN_NAME}")
  [[ -n "${WANDB_ENTITY:-}" ]] && EXTRA_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --targets-config "${TARGETS_CONFIG}" \
  --fasta-path "${FASTA_PATH:-/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa}" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --checkpoint-dir "checkpoints/v0data/${RUN_NAME}${RUN_SUFFIX:-}" \
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
  --base-param-dtype float32 \
  --lora-param-dtype float32 \
  --locon-param-dtype float32 \
  --activation-dtype bfloat16 \
  --base-compute-dtype bfloat16 \
  --lora-compute-dtype bfloat16 \
  --locon-compute-dtype bfloat16 \
  --best-metric "${BEST_METRIC}" \
  --best-metric-mode max \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-5}" \
  --eval-splits valid,test \
  --target-workers "${TARGET_WORKERS:-8}" \
  --window-workers "${WINDOW_WORKERS:-8}" \
  --prefetch-batches 2 \
  --progress-interval 100 \
  "${EXTRA_ARGS[@]}"
