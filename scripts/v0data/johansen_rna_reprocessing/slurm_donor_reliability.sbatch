#!/bin/bash
#SBATCH --job-name=johansen_rna_reliability
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/v0data/johansen_rna_reliability-%A_%a.out
#SBATCH --error=logs/v0data/johansen_rna_reliability-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data results
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

species_names=(human macaque marmoset)
file_prefixes=(Human Macaque Marmoset)
species="${species_names[$SLURM_ARRAY_TASK_ID]}"
prefix="${file_prefixes[$SLURM_ARRAY_TASK_ID]}"
rna_root=/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/AllenBrainMultiome/RNA

"${HOME}/venv/jax/bin/python" scripts/v0data/johansen_rna_reprocessing/audit_donor_reliability.py \
  --species "$species" \
  --expression "$rna_root/${prefix}_HMBA_basalganglia_AIT_pre-print.h5ad" \
  --template "outputs/v0data/johansen-fragment-joint-depth-filtered/$species/gene_expression_supervision.npz" \
  --orthologs "$rna_root/orthologs_1to1_final.csv" \
  --output "results/v0data_johansen_${species}_rna_donor_reliability.json"
