#!/bin/bash
# Submit depth-filtered gene-only RNA screens after preparing target manifests.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data outputs/v0data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python_bin="${HOME}/venv/jax/bin/python"
liu_root=outputs/v0data/liu-hdma-depth10m
"$python_bin" scripts/v0data/filter_target_groups_by_depth.py \
  --input outputs/v0data/liu-hdma/joint/targets.json \
  --depths outputs/v0data/liu-hdma/atac_totals.npz \
  --output "$liu_root/targets.json" \
  --gene-output "$liu_root/gene_expression_supervision.npz" \
  --minimum-fragments 10000000 \
  --atac-head liu_atac \
  --rna-head liu_rna
"$python_bin" scripts/v0data/prepare_gene_only_rna_config.py \
  --input "$liu_root/targets.json" \
  --output "$liu_root/targets_geneonly.json" \
  --head liu_rna
"$python_bin" scripts/v0data/prepare_gene_only_rna_config.py \
  --input "$liu_root/targets.json" \
  --output "$liu_root/targets_geneonly_corrw0p1.json" \
  --head liu_rna \
  --correlation-loss-weight 0.1

johansen_root=outputs/v0data/johansen-fragment-joint-depth10m
"$python_bin" scripts/allen_atac_reprocessing/prepare_multispecies_targets.py \
  --source-species-config outputs/allen_brain_multiome_multispecies_v1/species.json \
  --atac-root outputs/v0data/johansen-fragment-atac \
  --output-dir "$johansen_root" \
  --fragment-shards human=outputs/allen_atac_full_depth/shards \
  --fragment-shards macaque=outputs/v0data/johansen-fragment-atac/macaque/shards \
  --fragment-shards marmoset=outputs/v0data/johansen-fragment-atac/marmoset/shards \
  --minimum-fragments 10000000
"$python_bin" scripts/v0data/prepare_gene_only_species_config.py \
  --input "$johansen_root/species.json" \
  --output-dir "${johansen_root}-geneonly" \
  --head allen_rna
"$python_bin" scripts/v0data/prepare_gene_only_species_config.py \
  --input "$johansen_root/species.json" \
  --output-dir "${johansen_root}-geneonly-corrw0p1" \
  --head allen_rna \
  --correlation-loss-weight 0.1

submit_pair() {
  local launcher=$1
  local export_args=$2
  local smoke full
  smoke=$(sbatch --parsable --nice=5000 --array=0-1%2 --export="ALL,SMOKE=1,${export_args}" "$launcher")
  full=$(sbatch --parsable --nice=5000 --array=0-1%2 --dependency="afterok:${smoke}" --export="ALL,${export_args}" "$launcher")
  printf '%s smoke=%s full=%s\n' "$export_args" "$smoke" "$full"
}

submit_pair scripts/v0data/slurm_joint_adapter_comparison.sbatch \
  "DATASET=liu-hdma,TARGETS_CONFIG=${liu_root}/targets_geneonly.json,RUN_SUFFIX=_depth10m_geneonly"
submit_pair scripts/v0data/slurm_joint_adapter_comparison.sbatch \
  "DATASET=liu-hdma,TARGETS_CONFIG=${liu_root}/targets_geneonly_corrw0p1.json,RUN_SUFFIX=_depth10m_geneonly_corrw0p1"
submit_pair scripts/v0data/slurm_johansen_joint_adapters.sbatch \
  "SPECIES_CONFIG=${johansen_root}-geneonly/species.json,RUN_SUFFIX=_depth10m_geneonly"
submit_pair scripts/v0data/slurm_johansen_joint_adapters.sbatch \
  "SPECIES_CONFIG=${johansen_root}-geneonly-corrw0p1/species.json,RUN_SUFFIX=_depth10m_geneonly_corrw0p1"
