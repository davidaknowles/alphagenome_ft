#!/bin/bash
# Score one trained model on one eQTL record cache.
# Requires MODEL_ARG (label=checkpoint), OUTPUT_DIR, RECORDS_CACHE; optional
# BASE_CHECKPOINT (pretrained AlphaGenome weights) and DATASET_CONFIG (head layout).
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/eqtl/benchmark-%j.out
#SBATCH --error=logs/eqtl/benchmark-%j.err

set -euo pipefail
cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
source "$HOME/venv/jax/bin/activate"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
: "${MODEL_ARG:?MODEL_ARG must be label=checkpoint}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${RECORDS_CACHE:?RECORDS_CACHE is required}"
extra=()
if [[ -n "${BASE_CHECKPOINT:-}" ]]; then extra+=(--base-checkpoint "$BASE_CHECKPOINT"); fi
if [[ -n "${DATASET_CONFIG:-}" ]]; then extra+=(--dataset-config "$DATASET_CONFIG"); fi
mkdir -p "$OUTPUT_DIR"
exec python scripts/eqtl/benchmark.py \
  --output-dir "$OUTPUT_DIR" \
  --records-cache "$RECORDS_CACHE" \
  --max-per-panel 1000 --batch-size 2 --ref-batch-size 1 \
  --model "$MODEL_ARG" "${extra[@]}"
