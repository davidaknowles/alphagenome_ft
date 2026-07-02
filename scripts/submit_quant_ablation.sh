#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/venv/jax/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/quant_ablation/$STAMP}"
MERGED_CHECKPOINT="${MERGED_CHECKPOINT:-$OUTPUT_ROOT/merged_jax_default_lora_locon}"
TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/alphagenome_target_cache/humanbraindev_atac_w131072_float16}"

: "${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT to the finetuned adapter checkpoint to merge}"

mkdir -p "$OUTPUT_ROOT" logs/quant_ablation

if [[ ! -d "$MERGED_CHECKPOINT/checkpoint" ]]; then
  "$PYTHON" scripts/merge_finetuned_base.py \
    --source-checkpoint "$SOURCE_CHECKPOINT" \
    --output-checkpoint "$MERGED_CHECKPOINT"
fi

STRATEGY_FILE="$OUTPUT_ROOT/strategies.txt"
cat > "$STRATEGY_FILE" <<'STRATEGIES'
default
bf16_params
STRATEGIES

N_STRATEGIES=$(grep -cve '^[[:space:]]*$' "$STRATEGY_FILE")
if [[ "$N_STRATEGIES" -lt 1 ]]; then
  echo "No strategies found in $STRATEGY_FILE" >&2
  exit 1
fi

export ROOT PYTHON OUTPUT_ROOT STRATEGY_FILE
export CHECKPOINT="$MERGED_CHECKPOINT"
export TARGET_CACHE_DIR
export BATCH_SIZE="${BATCH_SIZE:-8}"
export TARGET_WORKERS="${TARGET_WORKERS:-12}"
export WINDOW_WORKERS="${WINDOW_WORKERS:-4}"
export SPLITS="${SPLITS:-valid,test}"

JOB_ID=$(sbatch --array "0-$((N_STRATEGIES - 1))" scripts/slurm_quant_ablation.sbatch | awk '{print $4}')
echo "submitted_job_id=$JOB_ID"
echo "output_root=$OUTPUT_ROOT"
echo "merged_checkpoint=$MERGED_CHECKPOINT"
echo "strategy_file=$STRATEGY_FILE"
