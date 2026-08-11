#!/bin/bash
#SBATCH --job-name=zemke23_joint
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/v0data/zemke23_joint-%A_%a.out
#SBATCH --error=logs/v0data/zemke23_joint-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data checkpoints/v0data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/outputs/jax_compilation_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

STRATEGIES=(lora lora+locon)
strategy="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"
run_name="zemke2023_joint_${strategy//+/_}${RUN_SUFFIX:-}"
species_config=outputs/v0data/zemke2023-species/species.json
test -f "$species_config"

EXTRA_ARGS=(--num-epochs "${NUM_EPOCHS:-100}")
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS=(--num-epochs 1 --max-train-steps 4 --limit-train 8 --limit-valid 8 --limit-test 8)
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  EXTRA_ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$run_name")
  [[ -n "${WANDB_ENTITY:-}" ]] && EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --species-config "$species_config" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --checkpoint-dir "checkpoints/v0data/${run_name}" \
  --window-size 131072 \
  --stride 131072 \
  --batch-size "${BATCH_SIZE:-8}" \
  --drop-last \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --backbone-lora \
  --adapter-strategy "$strategy" \
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
  --best-metric valid/mean/differential_pearson_r \
  --best-metric-mode max \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-5}" \
  --eval-splits valid,test \
  --target-workers "${TARGET_WORKERS:-8}" \
  --window-workers "${WINDOW_WORKERS:-8}" \
  --prefetch-batches 2 \
  --progress-interval 100 \
  "${EXTRA_ARGS[@]}"
