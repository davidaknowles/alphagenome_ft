#!/bin/bash
#SBATCH -J allen_atac_bw
#SBATCH -p cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/allen_atac_reprocess/%x-%j.out
#SBATCH -e logs/allen_atac_reprocess/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/allen_atac_reprocess

export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/allen_atac_reprocessing/materialize.py \
  --shard-dir outputs/allen_atac_full_depth/shards \
  --fai /gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa.fai \
  --output-dir outputs/allen_atac_full_depth/bigwigs \
  --field coverage_spmr
