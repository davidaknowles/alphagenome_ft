#!/bin/bash
#SBATCH --job-name=johansen_joint
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/v0data/johansen_joint-%A_%a.out
#SBATCH --error=logs/v0data/johansen_joint-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data checkpoints/v0data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/outputs/jax_compilation_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

STRATEGIES=(lora lora+locon)
strategy="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"
run_name="johansen_joint_${strategy//+/_}${RUN_SUFFIX:-}"
species_config="${SPECIES_CONFIG:-outputs/v0data/johansen-fragment-joint-depth-filtered/species.json}"
test -f "$species_config"

EXTRA_ARGS=(--num-epochs "${NUM_EPOCHS:-100}")
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS=(--num-epochs 1 --max-train-steps 3 --limit-train 4 --limit-valid 4 --limit-test 4)
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  EXTRA_ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$run_name")
  [[ -n "${WANDB_ENTITY:-}" ]] && EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  EXTRA_ARGS+=(--resume-from "${RESUME_FROM}")
fi
if [[ "${REPORT_HEAD_GRADIENT_NORMS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--report-head-gradient-norms)
fi

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --species-config "$species_config" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --checkpoint-dir "checkpoints/v0data/${run_name}" \
  --window-size 131072 \
  --stride 131072 \
  --batch-size "${BATCH_SIZE:-4}" \
  --drop-last \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --backbone-lora \
  --adapter-strategy "$strategy" \
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
  --best-metric valid/mean/differential_pearson_r \
  --best-metric-mode max \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-5}" \
  --eval-splits valid,test \
  --target-workers "${TARGET_WORKERS:-24}" \
  --window-workers "${WINDOW_WORKERS:-4}" \
  --prefetch-batches 1 \
  --progress-interval 100 \
  "${EXTRA_ARGS[@]}"
