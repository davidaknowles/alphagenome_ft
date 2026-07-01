#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
cd "$ROOT"

TORCH_REPO="${TORCH_REPO:-/gpfs/commons/home/daknowles/projects/alphagenome-pytorch}"
PYTHON="${PYTHON:-$HOME/venv/torchfix/bin/python}"
JAX_OUTPUT_ROOT="${JAX_OUTPUT_ROOT:-$ROOT/outputs/quant_ablation/20260628_212653}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/quant_ablation/$(date +%Y%m%d_%H%M%S)_torch}"
MERGED_JAX="${MERGED_JAX:-$JAX_OUTPUT_ROOT/merged_jax_default_lora_locon}"
JAX_CHECKPOINT="${JAX_CHECKPOINT:-$MERGED_JAX/checkpoint}"
TORCH_WEIGHTS="${TORCH_WEIGHTS:-$OUTPUT_ROOT/merged_torch_default_lora_locon.pth}"
TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/alphagenome_target_cache/humanbraindev_atac_w131072_float16}"

mkdir -p "$OUTPUT_ROOT" logs/quant_ablation

STRATEGY_FILE="$OUTPUT_ROOT/torch_strategies.txt"
cat > "$STRATEGY_FILE" <<'STRATEGIES'
default
bf16_params
torchao_float8_tower_linear
torchao_float8_linear
torchao_float8_all_linear
torchao_float8_linear_1x1conv
torchao_nvfp4_weight_only_tower_linear
torchao_nvfp4_weight_only_linear
torchao_nvfp4_weight_only_all_linear
torchao_nvfp4_weight_only_linear_1x1conv
bnb_nf4_weight_only_tower_linear
bnb_nf4_weight_only_linear
bnb_nf4_weight_only_all_linear
bnb_nf4_weight_only_linear_1x1conv
STRATEGIES

N_STRATEGIES=$(grep -cve '^[[:space:]]*$' "$STRATEGY_FILE")

BATCH_SIZE_FILE="$OUTPUT_ROOT/batch_sizes.txt"
printf '%s\n' "${BATCH_SIZES:-32}" | tr ',' '\n' | sed '/^[[:space:]]*$/d' > "$BATCH_SIZE_FILE"
N_BATCH_SIZES=$(grep -cve '^[[:space:]]*$' "$BATCH_SIZE_FILE")
if [[ "$N_BATCH_SIZES" -lt 1 ]]; then
  echo "No batch sizes found in $BATCH_SIZE_FILE" >&2
  exit 1
fi

export ROOT TORCH_REPO PYTHON JAX_CHECKPOINT TORCH_WEIGHTS
export BIGWIG_DIR="${BIGWIG_DIR:-/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs}"
CONVERT_JOB=$(sbatch scripts/slurm_convert_merged_to_torch.sbatch | awk '{print $4}')

export OUTPUT_ROOT STRATEGY_FILE BATCH_SIZE_FILE N_BATCH_SIZES TORCH_WEIGHTS TORCH_REPO TARGET_CACHE_DIR
export TARGET_WORKERS="${TARGET_WORKERS:-12}"
export WINDOW_WORKERS="${WINDOW_WORKERS:-4}"
export SPLITS="${SPLITS:-valid,test}"
export MAX_BATCHES="${MAX_BATCHES:-}"

N_TASKS=$((N_STRATEGIES * N_BATCH_SIZES))
ARRAY_JOB=$(sbatch --dependency "afterok:$CONVERT_JOB" --array "0-$((N_TASKS - 1))" scripts/slurm_torch_quant_ablation.sbatch | awk '{print $4}')

echo "convert_job_id=$CONVERT_JOB"
echo "torch_array_job_id=$ARRAY_JOB"
echo "output_root=$OUTPUT_ROOT"
echo "torch_weights=$TORCH_WEIGHTS"
echo "strategy_file=$STRATEGY_FILE"
echo "batch_size_file=$BATCH_SIZE_FILE"
