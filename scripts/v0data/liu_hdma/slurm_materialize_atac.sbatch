#!/bin/bash
#SBATCH --job-name=liu_atac_bigwig
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v0data/liu_atac_bigwig-%j.out
#SBATCH --error=logs/v0data/liu_atac_bigwig-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/liu-hdma/atac_bigwigs
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"

"${HOME}/venv/jax/bin/python" scripts/allen_atac_reprocessing/materialize.py \
  --shard-dir outputs/v0data/liu-hdma/atac_shards \
  --fai /gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa.fai \
  --output-dir outputs/v0data/liu-hdma/atac_bigwigs \
  --field coverage_spmr \
  --head-id liu_atac
