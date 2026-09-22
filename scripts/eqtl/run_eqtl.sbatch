#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/eqtl/eqtl-%j.out
#SBATCH --error=results/eqtl/eqtl-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/eqtl
source "$HOME/venv/jax/bin/activate"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
exec python scripts/eqtl/benchmark.py \
  --output-dir results/eqtl/finemap_1mb_trained_models \
  --max-per-panel 1000 \
  --batch-size 2 \
  --ref-batch-size 1 \
  --model head_only=checkpoints/v0data/joint_all_nonencode_all_gene_source_specific_semantic_neural_accessibility_bootstrap_head_warmup_seed_epoch12 \
  --model lora=checkpoints/v0data/joint_all_nonencode_lora_all_gene_source_specific_semantic_neural_accessibility_bootstrap_headwarm12_adapters_lr3e4_reset/best \
  --model locon=checkpoints/v0data/joint_all_nonencode_lora_locon_all_gene_source_specific_semantic_neural_accessibility_bootstrap_headwarm12_adapters_lr3e4_reset/best
