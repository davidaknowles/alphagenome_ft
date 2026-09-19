#!/bin/bash
# Short 1Mb base-model pilot for alternate-batch size with safe reference batching.
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=results/eqtl/alt-pilot-%A_%a.out
#SBATCH --error=results/eqtl/alt-pilot-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/eqtl
source "$HOME/venv/jax/bin/activate"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

BATCH_SIZE=(2 4 8)
batch="${BATCH_SIZE[$SLURM_ARRAY_TASK_ID]}"
out="results/eqtl/alt_pilot_b${batch}"
mkdir -p "$out"

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 2 > "${out}/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

python scripts/eqtl/benchmark.py \
  --output-dir "$out" \
  --records-cache results/eqtl/finemap_1mb/records.jsonl \
  --max-records 2000 \
  --batch-size "$batch" \
  --ref-batch-size 1 \
  --model base=base
