#!/bin/bash
# Screen row-centered RNA objective weights before matched LoCon training.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data/liu-hdma/joint
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
sbatch_bin="${SBATCH_BIN:-sbatch}"
weights=(0 0.1 1 10)
suffixes=(0 0p1 1 10)

"${HOME}/venv/jax/bin/python" \
  scripts/v0data/prepare_reconstructed_row_sweeps.py

for index in "${!weights[@]}"; do
  weight="${weights[$index]}"
  suffix="${suffixes[$index]}"

  liu_targets="outputs/v0data/liu-hdma/joint/targets_geneonly_rowcorrw${suffix}.json"
  test -f "$liu_targets"
  liu_exports="ALL,DATASET=liu-hdma,TARGETS_CONFIG=${liu_targets},RUN_SUFFIX=_geneonly_rowcorrw${suffix}_screen,NUM_EPOCHS=1,BALANCE_GENE_WINDOWS=1"
  liu_smoke=$(
    "$sbatch_bin" --parsable --nice="${NICE:-60}" --time=00:30:00 \
      --array=0 --export="${liu_exports},SMOKE=1" \
      scripts/v0data/slurm_joint_adapter_comparison.sbatch
  )
  liu_full=$(
    "$sbatch_bin" --parsable --nice="${NICE:-60}" --time=08:00:00 \
      --array=0 --dependency="afterok:${liu_smoke}_*" --export="$liu_exports" \
      scripts/v0data/slurm_joint_adapter_comparison.sbatch
  )

  johansen_root="outputs/v0data/johansen-rna-corrected/geneonly-rowcorrw${suffix}"
  test -f "$johansen_root/species.json"
  johansen_exports="ALL,SPECIES_CONFIG=${johansen_root}/species.json,RUN_SUFFIX=_geneonly_rowcorrw${suffix}_screen,NUM_EPOCHS=1,BALANCE_GENE_WINDOWS=1"
  johansen_smoke=$(
    "$sbatch_bin" --parsable --nice="${NICE:-60}" --gres=gpu:l40s:2 \
      --time=00:30:00 --array=0 --export="${johansen_exports},SMOKE=1" \
      scripts/v0data/slurm_johansen_joint_adapters.sbatch
  )
  johansen_full=$(
    "$sbatch_bin" --parsable --nice="${NICE:-60}" --gres=gpu:l40s:2 \
      --time=08:00:00 --array=0 --dependency="afterok:${johansen_smoke}_*" \
      --export="$johansen_exports" scripts/v0data/slurm_johansen_joint_adapters.sbatch
  )

  printf 'weight=%s liu_smoke=%s liu_full=%s johansen_smoke=%s johansen_full=%s\n' \
    "$weight" "$liu_smoke" "$liu_full" "$johansen_smoke" "$johansen_full"
done
