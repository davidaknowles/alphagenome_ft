#!/bin/bash
# Screen row-centered correlation on the published-track-integrated human target.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data/zemke2023-published-rowcorr
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
sbatch_bin="${SBATCH_BIN:-sbatch}"
base=outputs/v0data/zemke2023-published-gene-supervision/human/targets.json
weights_string="${ROW_WEIGHTS:-0;1;10}"
IFS=';' read -r -a weights <<< "$weights_string"

for weight in "${weights[@]}"; do
  suffix="${weight//./p}"
  targets="outputs/v0data/zemke2023-published-rowcorr/targets_rowcorrw${suffix}.json"
  "${HOME}/venv/jax/bin/python" scripts/v0data/prepare_head_correlation_config.py \
    --input "$base" \
    --output "$targets" \
    --head zemke2023_rna \
    --double-centered-weight 0 \
    --row-centered-weight "$weight"

  exports="ALL,TARGETS_CONFIG=${targets},RUN_SUFFIX=_published_gene_balanced_rowcorrw${suffix}_screen,NUM_EPOCHS=1,BALANCE_GENE_WINDOWS=1"
  smoke=$(
    "$sbatch_bin" --parsable --nice="${NICE:-90}" --time=00:30:00 \
      --array=0 --export="${exports},SMOKE=1" \
      scripts/v0data/slurm_zemke2023_adapter_matrix.sbatch
  )
  full=$(
    "$sbatch_bin" --parsable --nice="${NICE:-90}" --time=12:00:00 \
      --array=0 --dependency="afterok:${smoke}_*" --export="$exports" \
      scripts/v0data/slurm_zemke2023_adapter_matrix.sbatch
  )
  source="checkpoints/v0data/zemke2023_human_lora_published_gene_balanced_rowcorrw${suffix}_screen/best"
  evaluation=$(
    "$sbatch_bin" --parsable --nice="${NICE:-90}" --time=01:00:00 \
      --array=0 --dependency="afterok:${full}_0" \
      --export="ALL,TARGETS_CONFIG=outputs/v0data/zemke2023-species/human/targets.json,RESUME_FROM=${source},RUN_SUFFIX=_published_gene_balanced_rowcorrw${suffix}_screen_coordinate_eval,EVALUATE_ONLY=1" \
      scripts/v0data/slurm_zemke2023_adapter_matrix.sbatch
  )
  printf 'Zemke published row-correlation weight %s LoRA smoke=%s full=%s coordinate=%s\n' \
    "$weight" "$smoke" "$full" "$evaluation"
done
