#!/bin/bash
# Submit a matched one-epoch HDA rank-16 RNA-head screen.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data/hda-joint
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
sbatch_bin="${SBATCH_BIN:-sbatch}"

rank="${RNA_OUTPUT_RANK:-16}"
targets="outputs/v0data/hda-joint/targets_rna_rank${rank}.json"
"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_factorized_head_config.py \
  --input outputs/v0data/hda-joint/targets.json \
  --output "$targets" \
  --head hda_rna \
  --rank "$rank"

exports="ALL,DATASET=hda-joint,TARGETS_CONFIG=${targets},RUN_SUFFIX=_rna_rank${rank}_screen,NUM_EPOCHS=1"
smoke=$("$sbatch_bin" --parsable --nice="${NICE:-100}" --time=00:30:00 \
  --cpus-per-task=8 --array=0-1%2 --export="${exports},SMOKE=1" \
  scripts/v0data/slurm_joint_adapter_comparison.sbatch)
full=$("$sbatch_bin" --parsable --nice="${NICE:-100}" --time=12:00:00 \
  --array=0-1%2 --dependency="afterok:${smoke}_*" --export="$exports" \
  scripts/v0data/slurm_joint_adapter_comparison.sbatch)
printf 'HDA RNA rank %s smoke=%s full=%s\n' "$rank" "$smoke" "$full"
