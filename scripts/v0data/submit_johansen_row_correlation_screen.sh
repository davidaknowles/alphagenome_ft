#!/bin/bash
# Submit a matched corrected-Johansen row-centered RNA objective screen.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
sbatch_bin="${SBATCH_BIN:-sbatch}"

weight="${ROW_CORRELATION_LOSS_WEIGHT:-10}"
suffix="${weight//./p}"
source_config=outputs/v0data/johansen-rna-corrected/geneonly-corrw1/species.json
output_root="outputs/v0data/johansen-rna-corrected/geneonly-rowcorrw${suffix}"

"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_gene_only_species_config.py \
  --input "$source_config" \
  --output-dir "$output_root" \
  --head allen_rna \
  --row-correlation-loss-weight "$weight"

exports="ALL,SPECIES_CONFIG=${output_root}/species.json,RUN_SUFFIX=_geneonly_rowcorrw${suffix},NUM_EPOCHS=1,BALANCE_GENE_WINDOWS=1"
smoke=$("$sbatch_bin" --parsable --gres=gpu:l40s:2 --time=00:30:00 --cpus-per-task=8 \
  --array=0-1%2 --export="${exports},SMOKE=1" \
  scripts/v0data/slurm_johansen_joint_adapters.sbatch)
full=$("$sbatch_bin" --parsable --gres=gpu:l40s:2 --array=0-1%2 \
  --dependency="afterok:${smoke}_*" --export="$exports" \
  scripts/v0data/slurm_johansen_joint_adapters.sbatch)
printf 'johansen row-correlation weight %s smoke=%s full=%s\n' "$weight" "$smoke" "$full"
