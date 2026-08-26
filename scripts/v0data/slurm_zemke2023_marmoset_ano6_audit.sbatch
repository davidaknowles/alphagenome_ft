#!/bin/bash
#SBATCH --job-name=v0_z23_ano6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/v0data/v0_z23_ano6-%A_%a.out
#SBATCH --error=logs/v0data/v0_z23_ano6-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data evaluations/v0data/zemke2023_marmoset_ano6_audit
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/outputs/jax_compilation_cache}"

variants=(baseline exclude_rrna_locus)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
if [[ "$variant" == baseline ]]; then
  dataset_config="outputs/v0data/joint-objective-variants/metric-tempered/datasets.json"
else
  "${HOME}/venv/jax/bin/python" \
    scripts/v0data/zemke2023_rna_reprocessing/prepare_marmoset_ano6_audit.py \
    --dataset-config outputs/v0data/joint-objective-variants/metric-tempered/datasets.json \
    --supervision outputs/v0data/zemke2023-gene-supervision/marmoset/gene_expression_supervision.npz \
    --output-dir outputs/v0data/zemke2023-marmoset-ano6-audit
  dataset_config="outputs/v0data/zemke2023-marmoset-ano6-audit/datasets.json"
fi
source_checkpoint="${SOURCE_CHECKPOINT:-checkpoints/v0data/joint_all_nonencode_lora_locon_lr3e5_epoch31_metric_tempered_reset/best}"

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --dataset-config "$dataset_config" \
  --evaluate-only \
  --evaluate-source zemke2023:marmoset \
  --resume-from "$source_checkpoint" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --checkpoint-dir "evaluations/v0data/zemke2023_marmoset_ano6_audit/${variant}" \
  --split-source chromosome \
  --window-size 131072 \
  --stride 131072 \
  --batch-size 8 \
  --num-devices 1 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --backbone-lora \
  --adapter-strategy lora+locon \
  --lora-rank 16 \
  --lora-alpha 16 \
  --locon-rank 4 \
  --locon-alpha 1 \
  --activation-dtype bfloat16 \
  --backbone-compute-dtype bfloat16 \
  --base-compute-dtype bfloat16 \
  --lora-compute-dtype bfloat16 \
  --locon-compute-dtype bfloat16 \
  --target-workers 12 \
  --window-workers 4 \
  --eval-splits test
