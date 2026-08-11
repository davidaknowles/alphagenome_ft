#!/bin/bash
#SBATCH -J johansen_atac_bw
#SBATCH -p cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --array=0-2%3
#SBATCH -o logs/allen_atac_reprocess/%x-%A_%a.out
#SBATCH -e logs/allen_atac_reprocess/%x-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/allen_atac_reprocess
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

species=(human macaque marmoset)
species_name="${species[$SLURM_ARRAY_TASK_ID]}"
species_config=outputs/allen_brain_multiome_multispecies_v1/species.json
fasta="$(python scripts/allen_atac_reprocessing/species_config_query.py \
  --species-config "$species_config" \
  fasta \
  --species "$species_name")"
if [[ "$species_name" == human ]]; then
  shard_dir=outputs/allen_atac_full_depth/shards
else
  shard_dir="outputs/v0data/johansen-fragment-atac/$species_name/shards"
fi

python scripts/allen_atac_reprocessing/materialize.py \
  --shard-dir "$shard_dir" \
  --fai "${fasta}.fai" \
  --output-dir "outputs/v0data/johansen-fragment-atac/$species_name/bigwigs" \
  --field coverage_spmr \
  --head-id allen_atac
