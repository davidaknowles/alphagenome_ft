#!/bin/bash
# Stream XQTL parquets and broad singlebrain PIP tables for matched overlap.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/eqtl_xqtl/prepare-%j.out
#SBATCH --error=results/eqtl_xqtl/prepare-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/eqtl_xqtl/prepared
source "$HOME/venv/jax/bin/activate"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
exec python scripts/eqtl_xqtl/prepare.py --output-dir results/eqtl_xqtl/prepared
