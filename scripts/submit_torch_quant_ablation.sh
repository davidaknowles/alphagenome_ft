#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
cd "$ROOT"

TORCH_REPO="${TORCH_REPO:-/gpfs/commons/home/daknowles/projects/alphagenome-pytorch}"
PYTHON="${PYTHON:-$HOME/venv/torch/bin/python}"
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
nf4_linear_conservative
nf4_linear_aggressive
nf4_1x1conv
nf4_late_conv
nf4_all_conv
fp8_linear_conservative
fp8_linear_aggressive
nvfp4_weight_only
STRATEGIES

N_STRATEGIES=$(grep -cve '^[[:space:]]*$' "$STRATEGY_FILE")

export ROOT TORCH_REPO PYTHON JAX_CHECKPOINT TORCH_WEIGHTS
export BIGWIG_DIR="${BIGWIG_DIR:-/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs}"
CONVERT_JOB=$(sbatch scripts/slurm_convert_merged_to_torch.sbatch | awk '{print $4}')

export OUTPUT_ROOT STRATEGY_FILE TORCH_WEIGHTS TORCH_REPO TARGET_CACHE_DIR
export BATCH_SIZE="${BATCH_SIZE:-1}"
export TARGET_WORKERS="${TARGET_WORKERS:-12}"
export WINDOW_WORKERS="${WINDOW_WORKERS:-4}"
export SPLITS="${SPLITS:-valid,test}"

ARRAY_JOB=$(sbatch --dependency "afterok:$CONVERT_JOB" --array "0-$((N_STRATEGIES - 1))" scripts/slurm_torch_quant_ablation.sbatch | awk '{print $4}')

echo "convert_job_id=$CONVERT_JOB"
echo "torch_array_job_id=$ARRAY_JOB"
echo "output_root=$OUTPUT_ROOT"
echo "torch_weights=$TORCH_WEIGHTS"
echo "strategy_file=$STRATEGY_FILE"
