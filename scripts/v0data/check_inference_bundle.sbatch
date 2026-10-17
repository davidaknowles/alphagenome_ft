#!/bin/bash
# Compare an inference bundle against its source checkpoint. Set BUNDLE when submitting.
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=results/v0data/inference_bundle_check/check-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$HOME/venv/jax/bin/activate"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD"
export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
: "${BUNDLE:?BUNDLE must name the bundle directory}"
OUT=results/v0data/inference_bundle_check
CKPT=checkpoints/v0data/joint_all_nonencode_lora_locon_all_gene_source_specific_raw_zemke_atac_headwarm12_adapters_lr3e4_reset/best
CHECK=scripts/v0data/check_inference_bundle.py
python "$CHECK" --mode original --checkpoint "$CKPT" --output-dir "$OUT"
python "$CHECK" --mode original --checkpoint "$CKPT" --output-dir "$OUT" --name original_repeat
python "$CHECK" --mode bundle --bundle "$BUNDLE" --output-dir "$OUT"
python "$CHECK" --mode compare --output-dir "$OUT" --names original original_repeat
python "$CHECK" --mode compare --output-dir "$OUT" --names original bundle
