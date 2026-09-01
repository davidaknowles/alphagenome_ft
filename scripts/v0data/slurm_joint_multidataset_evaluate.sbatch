#!/bin/bash
#SBATCH --job-name=v0_joint_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/v0data/v0_joint_eval-%A_%a.out
#SBATCH --error=logs/v0data/v0_joint_eval-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data evaluations/v0data/joint_all_nonencode
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/outputs/jax_compilation_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

routes=(
  hda:human
  liu_hdma:human
  johansen2025:human
  johansen2025:macaque
  johansen2025:marmoset
  zemke2023:human
  zemke2023:macaque
  zemke2023:marmoset
  zemke2023:mouse
  zemke2024:human
)
strategies=(lora lora_locon)
if [[ -n "${EVALUATION_STRATEGY:-}" ]]; then
  strategy="$EVALUATION_STRATEGY"
  if [[ "$strategy" != lora && "$strategy" != lora_locon ]]; then
    printf 'Invalid evaluation strategy: %s\n' "$strategy" >&2
    exit 1
  fi
  if (( SLURM_ARRAY_TASK_ID >= ${#routes[@]} )); then
    printf 'Single-checkpoint task index is out of range: %s\n' "$SLURM_ARRAY_TASK_ID" >&2
    exit 1
  fi
  route_index=$SLURM_ARRAY_TASK_ID
  source_checkpoint="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
  evaluation_tag="${EVALUATION_TAG:?EVALUATION_TAG is required}"
  evaluation_base="${EVALUATION_ROOT:-evaluations/v0data/joint_all_nonencode}/${evaluation_tag}"
else
  route_index=$((SLURM_ARRAY_TASK_ID % ${#routes[@]}))
  strategy_index=$((SLURM_ARRAY_TASK_ID / ${#routes[@]}))
  strategy="${strategies[$strategy_index]}"
  run_suffix="${RUN_SUFFIX:-}"
  source_checkpoint="${CHECKPOINT_ROOT:-checkpoints/v0data}/joint_all_nonencode_${strategy}${run_suffix}/best"
  evaluation_base="${EVALUATION_ROOT:-evaluations/v0data/joint_all_nonencode}/${strategy}${run_suffix}"
fi
route="${routes[$route_index]}"
route_path="${route/:/_}"
evaluation_dir="${evaluation_base}/${route_path}"
dataset_config="${DATASET_CONFIG:-outputs/v0data/joint-all-nonencode/datasets.json}"
locon_targets="${LOCON_TARGETS:-default}"
locon_targets="${locon_targets//;/,}"
if [[ ! -f "$dataset_config" ]]; then
  printf 'Missing joint dataset configuration: %s\n' "$dataset_config" >&2
  exit 1
fi
if [[ ! -f "$source_checkpoint/metrics.json" ]]; then
  printf 'Missing source checkpoint metadata: %s\n' "$source_checkpoint/metrics.json" >&2
  exit 1
fi

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --dataset-config "$dataset_config" \
  --evaluate-only \
  --evaluate-source "$route" \
  --resume-from "$source_checkpoint" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --checkpoint-dir "$evaluation_dir" \
  --split-source chromosome \
  --window-size 131072 \
  --stride 131072 \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-devices 2 \
  --drop-last \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --backbone-lora \
  --adapter-strategy "${strategy/_/+}" \
  --lora-rank "${LORA_RANK:-16}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --locon-rank "${LOCON_RANK:-4}" \
  --locon-alpha "${LOCON_ALPHA:-1}" \
  --locon-targets "$locon_targets" \
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
  --eval-splits valid,test
