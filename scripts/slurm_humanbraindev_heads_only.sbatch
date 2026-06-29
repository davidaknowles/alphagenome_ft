#!/bin/bash
#SBATCH -J ag_hbd_atac
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o logs/humanbraindev_atac_heads_only/%x-%j.out
#SBATCH -e logs/humanbraindev_atac_heads_only/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/humanbraindev_atac_heads_only checkpoints/humanbraindev_atac_heads_only

export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "${HOME}/.kaggle/access_token" ]]; then
  export KAGGLE_API_TOKEN="$(< "${HOME}/.kaggle/access_token")"
fi
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-daknowles}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

if [[ -z "${CHECKPOINT_PATH:-}" && ! -f "${HOME}/.kaggle/access_token" && -z "${KAGGLE_API_TOKEN:-}" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
  echo "No CHECKPOINT_PATH or Kaggle token found; cannot load base AlphaGenome weights." >&2
  echo "Set CHECKPOINT_PATH=/path/to/local/alphagenome/checkpoint or provide ~/.kaggle/access_token." >&2
  exit 2
fi

python scripts/run_humanbraindev_finetune.py \
  --bigwig-dir "${BIGWIG_DIR:-/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs}" \
  --fasta-path "${FASTA_PATH:-/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa}" \
  --checkpoint-dir "${CHECKPOINT_DIR:-checkpoints/humanbraindev_atac_heads_only}" \
  --split-source "${SPLIT_SOURCE:-chromosome}" \
  --window-size "${WINDOW_SIZE:-131072}" \
  --stride "${STRIDE:-131072}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-epochs "${NUM_EPOCHS:-5}" \
  --base-param-dtype "${BASE_PARAM_DTYPE:-float32}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-2}" \
  --num-devices "${NUM_DEVICES:-1}" \
  ${MAX_TRAIN_STEPS:+--max-train-steps "$MAX_TRAIN_STEPS"} \
  ${LIMIT_TRAIN:+--limit-train "$LIMIT_TRAIN"} \
  ${LIMIT_VALID:+--limit-valid "$LIMIT_VALID"} \
  ${LIMIT_TEST:+--limit-test "$LIMIT_TEST"} \
  ${CHECKPOINT_PATH:+--checkpoint-path "$CHECKPOINT_PATH"} \
  ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"} \
  ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
  ${WANDB_RUN_NAME:+--wandb-run-name "$WANDB_RUN_NAME"} \
  ${WANDB_GROUP:+--wandb-group "$WANDB_GROUP"} \
  ${WANDB_TAGS:+--wandb-tags "$WANDB_TAGS"} \
  ${WANDB_JOB_TYPE:+--wandb-job-type "$WANDB_JOB_TYPE"} \
  ${WANDB_MODE:+--wandb-mode "$WANDB_MODE"}
