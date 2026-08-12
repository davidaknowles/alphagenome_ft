#!/bin/bash
#SBATCH --job-name=zemke24_gene_targets
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v0data/zemke24_gene_targets-%j.out
#SBATCH --error=logs/v0data/zemke24_gene_targets-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate

python scripts/v0data/zemke2024_rna_reprocessing/prepare_gene_supervision.py \
  --matrix-root /gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/zemke2024Epigenetic/RNAmatrix \
  --metadata outputs/v0data/zemke2024-gene-supervision/source/GSE278576_hippocampus_RNA_seurat_object_filtered_cells_metadata.tsv.gz \
  --targets outputs/v0data/zemke2024-all/targets.json \
  --gtf /gpfs/commons/home/daknowles/knowles_lab/index/hg38/gencode.v38.annotation.gtf.gz \
  --fasta /gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa \
  --output-dir outputs/v0data/zemke2024-gene-supervision \
  --correlation-loss-weight 10
