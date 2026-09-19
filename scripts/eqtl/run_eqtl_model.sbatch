#!/bin/bash
# Run exactly one model so base, LoRA, and LoCon can occupy separate GPUs.
# Set MODEL_LABEL to base, lora, or locon when submitting.
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/eqtl/parallel-%j.out
#SBATCH --error=results/eqtl/parallel-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/eqtl results/eqtl/finemap_1mb_parallel
source "$HOME/venv/jax/bin/activate"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

: "${MODEL_LABEL:?MODEL_LABEL must be base, lora, or locon}"
case "$MODEL_LABEL" in
  base)
    MODEL_ARG="base=base"
    ;;
  lora)
    MODEL_ARG="lora=checkpoints/v0data/joint_all_nonencode_lora_all_gene_source_specific_semantic_neural_accessibility_bootstrap_headwarm12_adapters_lr3e4_reset/best"
    ;;
  locon)
    MODEL_ARG="locon=checkpoints/v0data/joint_all_nonencode_lora_locon_all_gene_source_specific_semantic_neural_accessibility_bootstrap_headwarm12_adapters_lr3e4_reset/best"
    ;;
  *)
    echo "unsupported MODEL_LABEL=$MODEL_LABEL" >&2
    exit 2
    ;;
esac

exec python scripts/eqtl/benchmark.py \
  --output-dir results/eqtl/finemap_1mb_parallel \
  --records-cache results/eqtl/finemap_1mb/records.jsonl \
  --max-per-panel 1000 \
  --batch-size 2 \
  --ref-batch-size 1 \
  --model "$MODEL_ARG"
