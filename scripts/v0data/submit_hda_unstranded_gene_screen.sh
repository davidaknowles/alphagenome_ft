#!/bin/bash
# Compare a single unstranded RNA output per Mannens cell group with paired strand outputs.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data/hda-joint
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
sbatch_bin="${SBATCH_BIN:-sbatch}"

for layout in stranded unstranded; do
  targets="outputs/v0data/hda-joint/targets_geneonly_${layout}.json"
  extra_args=()
  if [[ "$layout" == "unstranded" ]]; then
    extra_args+=(--unstranded-output)
  fi
  "${HOME}/venv/jax/bin/python" scripts/v0data/prepare_gene_only_rna_config.py \
    --input outputs/v0data/hda-joint/targets.json \
    --output "$targets" \
    --head hda_rna \
    "${extra_args[@]}"

  exports="ALL,DATASET=hda-joint,TARGETS_CONFIG=${targets},RUN_SUFFIX=_geneonly_${layout}_balanced_screen,NUM_EPOCHS=1,BALANCE_GENE_WINDOWS=1"
  smoke=$("$sbatch_bin" --parsable --nice="${NICE:-10}" --time=00:30:00 \
    --cpus-per-task=8 --array=0 --export="${exports},SMOKE=1" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch)
  full=$("$sbatch_bin" --parsable --nice="${NICE:-10}" --time=12:00:00 \
    --array=0 --dependency="afterok:${smoke}_*" --export="$exports" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch)
  printf 'HDA %s gene-output LoRA smoke=%s full=%s\n' "$layout" "$smoke" "$full"
done
