#!/bin/bash
# Submit a LoRA screen assigning each Liu gene to its best exon-bearing window.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
targets=outputs/v0data/liu-hdma/joint/targets_geneonly_corrw1_exonwindow.json

"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_gene_window_assignment.py \
  --input outputs/v0data/liu-hdma/joint/targets_geneonly_corrw1.json \
  --output "$targets" \
  --head liu_rna \
  --assignment max_exon_overlap_scaled

common="DATASET=liu-hdma,TARGETS_CONFIG=${targets},RUN_SUFFIX=_exonwindow_screen,NUM_DEVICES=2"
smoke=$(
  "$sbatch_bin" --parsable --nice="${NICE:-20}" --gres=gpu:l40s:2 \
    --time=00:30:00 --array=0 \
    --export="ALL,${common},SMOKE=1" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch
)
full=$(
  "$sbatch_bin" --parsable --nice="${NICE:-20}" --gres=gpu:l40s:2 \
    --array=0 --dependency="afterok:${smoke}_*" \
    --export="ALL,${common},NUM_EPOCHS=1,EARLY_STOPPING_PATIENCE=0" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch
)
expanded_eval=$(
  "$sbatch_bin" --parsable --nice="${NICE:-20}" --gres=gpu:l40s:2 \
    --array=0 --dependency="afterok:${full}_*" \
    --export="ALL,DATASET=liu-hdma,TARGETS_CONFIG=${targets},RUN_SUFFIX=_canonical_exonwindow_eval,NUM_DEVICES=2,EVALUATE_ONLY=1,RESUME_FROM=checkpoints/v0data/liu-hdma_lora_geneonly_corrw1/best" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch
)
canonical_eval=$(
  "$sbatch_bin" --parsable --nice="${NICE:-20}" --gres=gpu:l40s:2 \
    --array=0 --dependency="afterok:${full}_*" \
    --export="ALL,DATASET=liu-hdma,TARGETS_CONFIG=outputs/v0data/liu-hdma/joint/targets_geneonly_corrw1.json,RUN_SUFFIX=_exonwindow_train_fullspan_eval,NUM_DEVICES=2,EVALUATE_ONLY=1,RESUME_FROM=checkpoints/v0data/liu-hdma_lora_exonwindow_screen/best" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch
)
printf 'liu_exon_window_lora_smoke=%s full=%s canonical_on_expanded=%s exon_trained_on_canonical=%s\n' \
  "$smoke" "$full" "$expanded_eval" "$canonical_eval"
