#!/bin/bash
# Short trained head-only pilot for 1Mb batch-size and peak-VRAM selection.
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=results/eqtl/pilot-%A_%a.out
#SBATCH --error=results/eqtl/pilot-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/eqtl
source "$HOME/venv/jax/bin/activate"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

BATCH_SIZE=(4 8 16)
batch="${BATCH_SIZE[$SLURM_ARRAY_TASK_ID]}"
out="results/eqtl/pilot_b${batch}"
mkdir -p "$out"

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 2 > "${out}/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

python scripts/eqtl/benchmark.py \
  --output-dir "$out" \
  --records-cache results/eqtl/finemap_1mb/records.jsonl \
  --max-records 128 \
  --batch-size "$batch" \
  --ref-batch-size "$batch" \
  --model head_only=checkpoints/v0data/joint_all_nonencode_all_gene_source_specific_semantic_neural_accessibility_bootstrap_head_warmup_seed_epoch12
