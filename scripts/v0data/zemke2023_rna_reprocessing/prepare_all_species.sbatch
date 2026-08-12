#!/bin/bash
#SBATCH --job-name=zemke23_gene_targets
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v0data/zemke23_gene_targets-%A_%a.out
#SBATCH --error=logs/v0data/zemke23_gene_targets-%A_%a.err
#SBATCH --array=0-3%2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

species=(human macaque marmoset mouse)
assemblies=(hg38 rheMac10 calJac4 mm10)
fastas=(
  /gpfs/commons/groups/knowles_lab/index/hg38/hg38.fa
  outputs/v0data/zemke2023-species/references/rheMac10.fa
  outputs/v0data/zemke2023-species/references/calJac4.fa
  outputs/v0data/zemke2023-species/references/mm10.fa
)
index="${SLURM_ARRAY_TASK_ID}"
name="${species[$index]}"
assembly="${assemblies[$index]}"
prefix="GSE229169_${name}_filtered"
study_root="/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/zemke2023Conserved"
expression_root="$study_root/cell_expression/by_species/$name"

"${HOME}/venv/jax/bin/python" scripts/v0data/zemke2023_rna_reprocessing/prepare_gene_supervision.py \
  --matrix "$expression_root/${prefix}_matrix.mtx.gz" \
  --barcodes "$expression_root/${prefix}_barcodes.tsv.gz" \
  --features "$expression_root/${prefix}_features.tsv.gz" \
  --metadata "$expression_root/${prefix}_metadata.tsv.gz" \
  --targets "outputs/v0data/zemke2023-species/$name/targets.json" \
  --gtf "outputs/v0data/zemke2023-gene-supervision/annotations/${assembly}.ncbiRefSeq.gtf.gz" \
  --fasta "${fastas[$index]}" \
  --output-dir "outputs/v0data/zemke2023-gene-supervision/$name" \
  --species "$name"
