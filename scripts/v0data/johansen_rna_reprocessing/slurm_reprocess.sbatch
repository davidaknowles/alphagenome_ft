#!/bin/bash
#SBATCH --job-name=johansen_rna_reprocess
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v0data/johansen_rna_reprocess-%j.out
#SBATCH --error=logs/v0data/johansen_rna_reprocess-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/johansen-rna-corrected
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python_bin="${HOME}/venv/jax/bin/python"
rna_root=/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/AllenBrainMultiome/RNA
output_root=outputs/v0data/johansen-rna-corrected
marmoset_expression="$output_root/Marmoset_HMBA_basalganglia_pseudobulk_by_group_correctnorm.h5ad"

"$python_bin" scripts/v0data/johansen_rna_reprocessing/aggregate_corrected_pseudobulk.py \
  --input "$rna_root/Marmoset_HMBA_basalganglia_AIT_pre-print.h5ad" \
  --output "$marmoset_expression"

for species in human macaque marmoset; do
  case "$species" in
    human) expression="$rna_root/Human_HMBA_basalganglia_pseudobulk_by_group_correctnorm.h5ad" ;;
    macaque) expression="$rna_root/Macaque_HMBA_basalganglia_pseudobulk_by_group_correctnorm.h5ad" ;;
    marmoset) expression="$marmoset_expression" ;;
  esac
  "$python_bin" scripts/v0data/johansen_rna_reprocessing/replace_gene_supervision.py \
    --species "$species" \
    --expression "$expression" \
    --template "outputs/v0data/johansen-fragment-joint-depth-filtered/$species/gene_expression_supervision.npz" \
    --orthologs "$rna_root/orthologs_1to1_final.csv" \
    --output "$output_root/supervision/$species/gene_expression_supervision.npz"
done

"$python_bin" scripts/v0data/johansen_rna_reprocessing/rewrite_species_config.py \
  --input outputs/v0data/johansen-fragment-joint-depth-filtered/species.json \
  --supervision-root "$output_root/supervision" \
  --output-dir "$output_root/geneonly-corrw1" \
  --correlation-loss-weight 1
