#!/bin/bash
# Compute original AlphaGenome RNA-seq and CAGE variant effects.
# Override OUTPUT_DIR and MAX_RECORDS for a pilot run.
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/eqtl/native-%j.out
#SBATCH --error=results/eqtl/native-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/eqtl
source "$HOME/venv/jax/bin/activate"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

OUTPUT_DIR=${OUTPUT_DIR:-results/eqtl/native_1mb_all_tracks}
ARGS=(
  --records-cache results/eqtl/finemap_1mb/records.jsonl
  --output-dir "$OUTPUT_DIR"
  --window 1048576
  --progress-interval 250
)
if [[ -n "${MAX_RECORDS:-}" ]]; then
  ARGS+=(--max-records "$MAX_RECORDS")
fi

exec python scripts/eqtl/benchmark_native.py "${ARGS[@]}"
