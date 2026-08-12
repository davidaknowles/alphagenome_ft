#!/bin/bash
#SBATCH --job-name=mouse_gradient_gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/v0data/mouse_gradient_gate-%j.out
#SBATCH --error=logs/v0data/mouse_gradient_gate-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

"${HOME}/venv/jax/bin/python" scripts/v0data/assert_gradient_diagnostics.py \
  checkpoints/v0data/zemke2023_mouse_lora_organismfix_gradnorm_smoke/gradient_diagnostics.json \
  checkpoints/v0data/zemke2023_mouse_lora_locon_organismfix_gradnorm_smoke/gradient_diagnostics.json \
  --expected-head zemke2023_atac \
  --expected-head zemke2023_rna \
  --minimum-norm 1e-8
