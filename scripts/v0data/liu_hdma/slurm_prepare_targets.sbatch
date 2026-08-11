#!/bin/bash
#SBATCH --job-name=liu_targets
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/v0data/liu_targets-%j.out
#SBATCH --error=logs/v0data/liu_targets-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/liu-hdma/joint
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

"${HOME}/venv/jax/bin/python" scripts/v0data/liu_hdma/prepare_targets.py \
  --rna-pseudobulk outputs/v0data/liu-hdma/rna_pseudobulk.h5ad \
  --atac-targets outputs/v0data/liu-hdma/atac_bigwigs/targets.json \
  --gtf /gpfs/commons/home/daknowles/knowles_lab/index/hg38/gencode.v38.annotation.gtf.gz \
  --fasta /gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa \
  --output-dir outputs/v0data/liu-hdma/joint
