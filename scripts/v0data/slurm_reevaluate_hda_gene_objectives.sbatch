#!/bin/bash
#SBATCH --job-name=hda_objective_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/v0data/hda_objective_eval-%A_%a.out
#SBATCH --error=logs/v0data/hda_objective_eval-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/outputs/jax_compilation_cache}"

suffixes=(0 0p1 1 10)
suffix="${suffixes[$SLURM_ARRAY_TASK_ID]}"
targets="outputs/v0data/hda-joint/targets_geneonly_corrw${suffix}.json"
run="checkpoints/v0data/hda-joint_lora_geneonly_corrw${suffix}_screen"
test -f "$targets"
test -f "$run/last/metrics.json"

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --targets-config "$targets" \
  --fasta-path /gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa \
  --include-chroms chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX \
  --checkpoint-path /gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1 \
  --checkpoint-dir "$run" \
  --resume-from "$run/last" \
  --evaluate-only \
  --split-source chromosome \
  --window-size 131072 \
  --stride 131072 \
  --batch-size 8 \
  --num-devices 2 \
  --drop-last \
  --backbone-lora \
  --adapter-strategy lora \
  --lora-rank 16 \
  --lora-alpha 16 \
  --base-param-dtype float32 \
  --lora-param-dtype float32 \
  --activation-dtype bfloat16 \
  --backbone-compute-dtype bfloat16 \
  --base-compute-dtype bfloat16 \
  --lora-compute-dtype bfloat16 \
  --eval-splits valid,test \
  --target-workers 24 \
  --window-workers 8 \
  --prefetch-batches 2
