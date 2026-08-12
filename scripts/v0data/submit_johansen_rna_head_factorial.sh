#!/bin/bash
# Screen RNA strand sharing and rank-32 output factorization on joint Johansen data.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data/johansen-rna-corrected
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
sbatch_bin="${SBATCH_BIN:-sbatch}"
base="outputs/v0data/johansen-rna-corrected/geneonly-corrw1/species.json"
rank="${RNA_OUTPUT_RANK:-32}"

declare -a variants=(unstranded "rank${rank}" "unstranded_rank${rank}")
for variant in "${variants[@]}"; do
  output_root="outputs/v0data/johansen-rna-corrected/geneonly-corrw1-${variant}"
  extra_args=()
  if [[ "$variant" == unstranded* ]]; then
    extra_args+=(--unstranded-output)
  fi
  if [[ "$variant" == *"rank${rank}" ]]; then
    extra_args+=(--output-rank "$rank")
  fi
  "${HOME}/venv/jax/bin/python" scripts/v0data/prepare_gene_only_species_config.py \
    --input "$base" \
    --output-dir "$output_root" \
    --head allen_rna \
    --correlation-loss-weight 1 \
    "${extra_args[@]}"

  exports="ALL,SPECIES_CONFIG=${output_root}/species.json,RUN_SUFFIX=_rawcount_geneonly_corrw1_${variant}_screen,NUM_EPOCHS=1"
  smoke=$("$sbatch_bin" --parsable --nice="${NICE:-95}" --time=00:30:00 \
    --cpus-per-task=8 --array=0 --export="${exports},SMOKE=1" \
    scripts/v0data/slurm_johansen_joint_adapters.sbatch)
  full=$("$sbatch_bin" --parsable --nice="${NICE:-95}" --time=12:00:00 \
    --array=0 --dependency="afterok:${smoke}_*" --export="$exports" \
    scripts/v0data/slurm_johansen_joint_adapters.sbatch)
  printf 'Johansen RNA head %s LoRA smoke=%s full=%s\n' "$variant" "$smoke" "$full"
done
