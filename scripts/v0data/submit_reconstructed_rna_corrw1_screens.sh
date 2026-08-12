#!/bin/bash
# Submit matched raw-CPM correlation screens for reconstructed RNA targets.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python_bin="${HOME}/venv/jax/bin/python"
liu_targets=outputs/v0data/liu-hdma/joint/targets_geneonly_corrw1.json
"$python_bin" scripts/v0data/prepare_gene_only_rna_config.py \
  --input outputs/v0data/liu-hdma/joint/targets.json \
  --output "$liu_targets" \
  --head liu_rna \
  --correlation-loss-weight 1

johansen_root=outputs/v0data/johansen-fragment-joint-depth-filtered-geneonly-corrw1
"$python_bin" scripts/v0data/prepare_gene_only_species_config.py \
  --input outputs/v0data/johansen-fragment-joint-depth-filtered/species.json \
  --output-dir "$johansen_root" \
  --head allen_rna \
  --correlation-loss-weight 1

submit_pair() {
  local launcher=$1
  local export_args=$2
  local smoke full
  smoke=$(sbatch --parsable --nice=1000 --time=00:30:00 --array=0-1%2 \
    --export="ALL,SMOKE=1,${export_args}" "$launcher")
  full=$(sbatch --parsable --nice=1500 --array=0-1%2 \
    --dependency="afterok:${smoke}_*" --export="ALL,${export_args}" "$launcher")
  printf '%s smoke=%s full=%s\n' "$export_args" "$smoke" "$full"
}

submit_pair scripts/v0data/slurm_joint_adapter_comparison.sbatch \
  "DATASET=liu-hdma,TARGETS_CONFIG=${liu_targets},RUN_SUFFIX=_geneonly_corrw1"
submit_pair scripts/v0data/slurm_johansen_joint_adapters.sbatch \
  "SPECIES_CONFIG=${johansen_root}/species.json,RUN_SUFFIX=_geneonly_corrw1"
