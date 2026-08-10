#!/bin/bash
#SBATCH -J ag_allen_prep3sp
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/allen_brain_multiome/%x-%j.out
#SBATCH -e logs/allen_brain_multiome/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/allen_brain_multiome
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

DATA_ROOT="/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/AllenBrainMultiome"

python scripts/prepare_allen_multispecies.py \
  --data-root "${DATA_ROOT}" \
  --human-fasta /gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa \
  --human-gtf /gpfs/commons/home/daknowles/knowles_lab/index/hg38/gencode.v38.annotation.gtf.gz \
  --output-dir "${OUTPUT_DIR:-outputs/allen_brain_multiome_multispecies_v1}"
