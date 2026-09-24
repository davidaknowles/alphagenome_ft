#!/bin/bash
# Compute original AlphaGenome RNA-seq and CAGE VEPs for XQTL variants.
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/eqtl_xqtl/native-%j.out
#SBATCH --error=results/eqtl_xqtl/native-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/eqtl_xqtl
source "$HOME/venv/jax/bin/activate"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
exec python scripts/eqtl/benchmark_native.py \
  --records-cache results/eqtl_xqtl/prepared/records.jsonl \
  --output-dir results/eqtl_xqtl/native_tracks \
  --window 1048576 --progress-interval 250
