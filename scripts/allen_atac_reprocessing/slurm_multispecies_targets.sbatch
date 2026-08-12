#!/bin/bash
#SBATCH -J johansen_targets
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o logs/allen_atac_reprocess/%x-%j.out
#SBATCH -e logs/allen_atac_reprocess/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/allen_atac_reprocess
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/allen_atac_reprocessing/prepare_multispecies_targets.py \
  --source-species-config outputs/allen_brain_multiome_multispecies_v1/species.json \
  --atac-root outputs/v0data/johansen-fragment-atac \
  --output-dir outputs/v0data/johansen-fragment-joint-depth-filtered \
  --fragment-shards human=outputs/allen_atac_full_depth/shards \
  --fragment-shards macaque=outputs/v0data/johansen-fragment-atac/macaque/shards \
  --fragment-shards marmoset=outputs/v0data/johansen-fragment-atac/marmoset/shards \
  --minimum-fragments 500000
