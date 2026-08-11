#!/bin/bash
#SBATCH -J allen_atac_pileup_ft
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/allen_atac_reprocess/%x-%j.out
#SBATCH -e logs/allen_atac_reprocess/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/allen_atac_reprocess checkpoints/allen_atac_full_depth_pileup

export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async

python scripts/run_humanbraindev_finetune.py \
  --targets-config outputs/allen_atac_full_depth/bigwigs/targets.json \
  --fasta-path /gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa \
  --checkpoint-path /gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1 \
  --checkpoint-dir checkpoints/allen_atac_full_depth_pileup \
  --split-source chromosome \
  --window-size 131072 \
  --stride 131072 \
  --batch-size 8 \
  --num-epochs 100 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --backbone-lora \
  --adapter-strategy lora \
  --base-param-dtype float32 \
  --lora-param-dtype float32 \
  --activation-dtype bfloat16 \
  --base-compute-dtype bfloat16 \
  --lora-compute-dtype bfloat16 \
  --best-metric valid_differential_pearson_r \
  --best-metric-mode max \
  --early-stopping-patience 3 \
  --eval-splits valid,test \
  --target-workers 32 \
  --prefetch-batches 1 \
  --wandb-project alphagenome-allen-brain \
  --wandb-group allen-atac-reprocessing \
  --wandb-run-name allen_atac_full_depth_fragment_pileup \
  --wandb-mode online
