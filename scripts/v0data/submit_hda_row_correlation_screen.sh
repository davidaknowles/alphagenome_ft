#!/bin/bash
# Submit a matched one-epoch Mannens row-centered RNA objective screen.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data/hda-joint
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
sbatch_bin="${SBATCH_BIN:-sbatch}"

weight="${ROW_CORRELATION_LOSS_WEIGHT:-10}"
tasks="${TASKS:-0-1%2}"
suffix="${weight//./p}"
targets="outputs/v0data/hda-joint/targets_geneonly_rowcorrw${suffix}.json"
"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_gene_only_rna_config.py \
  --input outputs/v0data/hda-joint/targets.json \
  --output "$targets" \
  --head hda_rna \
  --row-correlation-loss-weight "$weight"

exports="ALL,DATASET=hda-joint,TARGETS_CONFIG=${targets},RUN_SUFFIX=_geneonly_rowcorrw${suffix}_screen,NUM_EPOCHS=1,BALANCE_GENE_WINDOWS=1"
smoke=$("$sbatch_bin" --parsable --time=00:30:00 --cpus-per-task=8 --array="$tasks" \
  --export="${exports},SMOKE=1" scripts/v0data/slurm_joint_adapter_comparison.sbatch)
full=$("$sbatch_bin" --parsable --time=08:00:00 --array="$tasks" \
  --dependency="afterok:${smoke}_*" --export="$exports" \
  scripts/v0data/slurm_joint_adapter_comparison.sbatch)
printf 'hda row-correlation weight %s smoke=%s full=%s\n' "$weight" "$smoke" "$full"
