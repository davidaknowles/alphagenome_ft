#!/bin/bash
#SBATCH --job-name=hda_joint_targets
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/v0data/hda_joint_targets-%j.out
#SBATCH --error=logs/v0data/hda_joint_targets-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/hda-joint
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_hda_joint_targets.py \
  --rna-h5ad /gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/anndata/rna_pseudobulk.h5ad \
  --atac-bigwig-dir /gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs \
  --gtf /gpfs/commons/home/daknowles/knowles_lab/index/hg38/gencode.v38.annotation.gtf.gz \
  --fasta /gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa \
  --output-dir outputs/v0data/hda-joint
