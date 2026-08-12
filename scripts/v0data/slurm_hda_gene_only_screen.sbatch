#!/bin/bash
#SBATCH --job-name=hda_gene_only
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/v0data/hda_gene_only-%A_%a.out
#SBATCH --error=logs/v0data/hda_gene_only-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/hda-joint checkpoints/v0data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
weights=(0 0.1 1)
suffixes=(0 0p1 1)
screen_index="${SLURM_ARRAY_TASK_ID}"
weight="${weights[$screen_index]}"
suffix="${suffixes[$screen_index]}"
targets="outputs/v0data/hda-joint/targets_geneonly_corrw${suffix}.json"

"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_gene_only_rna_config.py \
  --input outputs/v0data/hda-joint/targets.json \
  --output "$targets" \
  --head hda_rna \
  --correlation-loss-weight "$weight"

export DATASET=hda-joint
export TARGETS_CONFIG="$targets"
export NUM_EPOCHS=1
export RUN_SUFFIX="_geneonly_corrw${suffix}_screen"
export SLURM_ARRAY_TASK_ID=0
bash scripts/v0data/slurm_joint_adapter_comparison.sbatch
