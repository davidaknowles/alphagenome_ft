#!/bin/bash
# Select the HDA LoRA objective and submit one matched LoRA plus LoCon epoch.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p results logs/v0data
sbatch_bin="${SBATCH_BIN:-sbatch}"
selection=results/v0data_hda_gene_objective_selection.json

"${HOME}/venv/jax/bin/python" scripts/v0data/select_hda_gene_objective.py \
  --minimum-improvement "${MINIMUM_IMPROVEMENT:-0}" \
  --json-output "$selection" \
  --markdown-output results/v0data_hda_gene_objective_selection.md

index=$(jq -r '.selected.index // empty' "$selection")
if [[ -z "$index" ]]; then
  echo "No nonzero HDA gene objective advanced."
  exit 0
fi
job=$(
  "$sbatch_bin" --parsable --array="$index" \
    --export=ALL,ADAPTER_TASK_ID=1 scripts/v0data/slurm_hda_gene_only_screen.sbatch
)
printf 'hda_gene_objective_lora_locon=%s\n' "$job"
