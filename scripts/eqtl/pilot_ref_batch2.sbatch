#!/bin/bash
# Check whether two 1Mb head-only reference windows fit while retaining alternate batch 8.
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/eqtl/ref2-pilot-%j.out
#SBATCH --error=results/eqtl/ref2-pilot-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/eqtl/ref2_pilot
source "$HOME/venv/jax/bin/activate"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 2 > results/eqtl/ref2_pilot/gpu.csv &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

python scripts/eqtl/benchmark.py \
  --output-dir results/eqtl/ref2_pilot \
  --records-cache results/eqtl/finemap_1mb/records.jsonl \
  --max-records 128 \
  --batch-size 8 \
  --ref-batch-size 2 \
  --model head_only=checkpoints/v0data/joint_all_nonencode_all_gene_source_specific_semantic_neural_accessibility_bootstrap_head_warmup_seed_epoch12
