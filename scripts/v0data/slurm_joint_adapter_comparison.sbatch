#!/bin/bash
#SBATCH --job-name=v0_joint_adapters
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/v0data/v0_joint_adapters-%A_%a.out
#SBATCH --error=logs/v0data/v0_joint_adapters-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data checkpoints/v0data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/outputs/jax_compilation_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

: "${DATASET:?Set DATASET to the target dataset name}"
STRATEGIES=(lora lora+locon)
strategy="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"
default_targets="outputs/v0data/${DATASET}/targets.json"
if [[ "$DATASET" == "liu-hdma" ]]; then
  default_targets="outputs/v0data/liu-hdma/joint/targets.json"
fi
targets="${TARGETS_CONFIG:-$default_targets}"
run_name="${DATASET}_${strategy//+/_}${RUN_SUFFIX:-}"
if [[ "${SMOKE:-0}" == "1" && "$run_name" != *_smoke ]]; then
  run_name="${run_name}_smoke"
fi
num_devices="${NUM_DEVICES:-${SLURM_GPUS_ON_NODE:-1}}"
canonical_chroms=chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX
test -f "$targets"

EXTRA_ARGS=(--num-epochs "${NUM_EPOCHS:-100}")
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS=(--num-epochs 1 --max-train-steps 1 --limit-train 16 --limit-valid 16 --limit-test 16)
fi
if (( num_devices > 1 )); then
  EXTRA_ARGS+=(--drop-last)
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  EXTRA_ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$run_name")
  [[ -n "${WANDB_ENTITY:-}" ]] && EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "${CORRELATION_LOSS_WEIGHT:-}" ]]; then
  EXTRA_ARGS+=(
    --double-centered-correlation-loss-weight "${CORRELATION_LOSS_WEIGHT}"
  )
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  EXTRA_ARGS+=(--resume-from "${RESUME_FROM}")
fi
if [[ "${REPORT_HEAD_GRADIENT_NORMS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--report-head-gradient-norms)
fi
if [[ "${BALANCE_GENE_WINDOWS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--balance-gene-windows)
fi

printf 'Using %s JAX device(s) with global batch size %s.\n' "$num_devices" "${BATCH_SIZE:-8}"

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --targets-config "$targets" \
  --fasta-path "${FASTA_PATH:-/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa}" \
  --include-chroms "$canonical_chroms" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --checkpoint-dir "checkpoints/v0data/${run_name}" \
  --pretrained-head-initialization "${PRETRAINED_HEAD_INITIALIZATION:-none}" \
  --split-source chromosome \
  --window-size 131072 \
  --stride 131072 \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-devices "$num_devices" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --learning-rate-schedule "${LEARNING_RATE_SCHEDULE:-constant}" \
  --warmup-steps "${WARMUP_STEPS:-0}" \
  --minimum-learning-rate-ratio "${MINIMUM_LEARNING_RATE_RATIO:-0.1}" \
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
  --window-workers "${WINDOW_WORKERS:-8}" \
  --prefetch-batches 2 \
  --progress-interval 100 \
  "${EXTRA_ARGS[@]}"
