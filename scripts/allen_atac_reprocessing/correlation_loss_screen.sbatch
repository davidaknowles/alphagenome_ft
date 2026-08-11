#!/bin/bash
#SBATCH -J allen_corr_loss
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --array=0-2%3
#SBATCH -o logs/allen_atac_reprocess/%x-%A_%a.out
#SBATCH -e logs/allen_atac_reprocess/%x-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/allen_atac_reprocess checkpoints/allen_atac_correlation_loss

export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async

weights=(1 10 100)
weight="${weights[$SLURM_ARRAY_TASK_ID]}"
run_name="weight${weight}${RUN_SUFFIX:-}"
extra_args=(--num-epochs 1)
if [[ "${SMOKE:-0}" == "1" ]]; then
  extra_args+=(--max-train-steps 1 --limit-train 8 --limit-valid 8 --limit-test 8)
fi

python scripts/run_humanbraindev_finetune.py \
  --targets-config outputs/allen_atac_full_depth/bigwigs/targets.json \
  --fasta-path /gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa \
  --checkpoint-path /gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1 \
  --checkpoint-dir "checkpoints/allen_atac_correlation_loss/${run_name}" \
  --split-source chromosome \
  --window-size 131072 \
  --stride 131072 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --double-centered-correlation-loss-weight "$weight" \
  --backbone-lora \
  --adapter-strategy lora \
  --base-param-dtype float32 \
  --lora-param-dtype float32 \
  --activation-dtype bfloat16 \
  --base-compute-dtype bfloat16 \
  --lora-compute-dtype bfloat16 \
  --best-metric valid/differential_pearson_r \
  --best-metric-mode max \
  --eval-splits valid,test \
  --target-workers 16 \
  --window-workers 8 \
  --prefetch-batches 1 \
  --progress-interval 100 \
  "${extra_args[@]}"
