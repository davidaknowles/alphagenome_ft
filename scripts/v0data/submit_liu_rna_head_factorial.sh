#!/bin/bash
# Screen RNA strand sharing and rank-16 output factorization on Liu HDMA.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data/liu-hdma/joint
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
sbatch_bin="${SBATCH_BIN:-sbatch}"
base="outputs/v0data/liu-hdma/joint/targets_geneonly_corrw1.json"
rank="${RNA_OUTPUT_RANK:-16}"

declare -a variants=(unstranded "rank${rank}" "unstranded_rank${rank}")
for variant in "${variants[@]}"; do
  targets="outputs/v0data/liu-hdma/joint/targets_geneonly_corrw1_${variant}.json"
  if [[ "$variant" == "rank${rank}" ]]; then
    "${HOME}/venv/jax/bin/python" scripts/v0data/prepare_factorized_head_config.py \
      --input "$base" \
      --output "$targets" \
      --head liu_rna \
      --rank "$rank"
  else
    extra_args=(--unstranded-output)
    if [[ "$variant" == "unstranded_rank${rank}" ]]; then
      extra_args+=(--output-rank "$rank")
    fi
    "${HOME}/venv/jax/bin/python" scripts/v0data/prepare_gene_only_rna_config.py \
      --input "$base" \
      --output "$targets" \
      --head liu_rna \
      --correlation-loss-weight 1 \
      "${extra_args[@]}"
  fi

  exports="ALL,DATASET=liu-hdma,TARGETS_CONFIG=${targets},RUN_SUFFIX=_geneonly_corrw1_${variant}_screen,NUM_EPOCHS=1"
  smoke=$("$sbatch_bin" --parsable --nice="${NICE:-90}" --time=00:30:00 \
    --cpus-per-task=8 --array=0 --export="${exports},SMOKE=1" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch)
  full=$("$sbatch_bin" --parsable --nice="${NICE:-90}" --time=12:00:00 \
    --array=0 --dependency="afterok:${smoke}_*" --export="$exports" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch)
  printf 'Liu RNA head %s LoRA smoke=%s full=%s\n' "$variant" "$smoke" "$full"
done
