#!/bin/bash
#SBATCH --job-name=v0_joint_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/v0data/v0_joint_all-%A_%a.out
#SBATCH --error=logs/v0data/v0_joint_all-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data checkpoints/v0data outputs/v0data/joint-all-nonencode
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/outputs/jax_compilation_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

strategies=(lora lora+locon)
strategy="${strategies[$SLURM_ARRAY_TASK_ID]}"
dataset_config="${DATASET_CONFIG:-outputs/v0data/joint-all-nonencode/datasets.json}"
run_name="joint_all_nonencode_${strategy//+/_}${RUN_SUFFIX:-}"
if [[ "${SMOKE:-0}" == "1" && "$run_name" != *_smoke ]]; then
  run_name="${run_name}_smoke"
fi
test -f "$dataset_config"

extra_args=(
  --num-epochs "${NUM_EPOCHS:-100}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-5}"
  --defer-test-evaluation
  --balance-gene-windows
  --drop-last
)
if [[ "${SMOKE:-0}" == "1" ]]; then
  extra_args=(
    --num-epochs 1
    --max-train-steps 5
    --limit-train 8
    --limit-valid 8
    --limit-test 8
    --drop-last
  )
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  extra_args+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$run_name")
  [[ -n "${WANDB_ENTITY:-}" ]] && extra_args+=(--wandb-entity "$WANDB_ENTITY")
fi

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --dataset-config "$dataset_config" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --checkpoint-dir "checkpoints/v0data/${run_name}" \
  --split-source chromosome \
  --window-size 131072 \
  --stride 131072 \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-devices 2 \
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
  --target-workers "${TARGET_WORKERS:-24}" \
  --window-workers "${WINDOW_WORKERS:-8}" \
  --best-metric valid/mean/differential_pearson_r \
  --best-metric-mode max \
  --eval-splits valid,test \
  "${extra_args[@]}"
