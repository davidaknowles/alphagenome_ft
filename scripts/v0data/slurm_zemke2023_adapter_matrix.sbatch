#!/bin/bash
#SBATCH --job-name=zemke23_adapters
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/v0data/zemke23_adapters-%A_%a.out
#SBATCH --error=logs/v0data/zemke23_adapters-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data checkpoints/v0data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/outputs/jax_compilation_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

SPECIES=(human human macaque macaque marmoset marmoset mouse mouse)
STRATEGIES=(lora lora+locon lora lora+locon lora lora+locon lora lora+locon)
species="${SPECIES[$SLURM_ARRAY_TASK_ID]}"
strategy="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"
organism=HOMO_SAPIENS
case "$species" in
  human)
    fasta=/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa
    include_chroms=chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX
    ;;
  macaque)
    fasta=outputs/v0data/zemke2023-species/references/rheMac10.fa
    include_chroms=chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chrX
    ;;
  marmoset)
    fasta=outputs/v0data/zemke2023-species/references/calJac4.fa
    include_chroms=chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX
    ;;
  mouse)
    fasta=outputs/v0data/zemke2023-species/references/mm10.fa
    organism=MUS_MUSCULUS
    include_chroms=chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chrX
    ;;
esac
targets="${TARGETS_CONFIG:-outputs/v0data/zemke2023-${species}/targets.json}"
run_name="zemke2023_${species}_${strategy//+/_}${RUN_SUFFIX:-}"
if [[ "${SMOKE:-0}" == "1" && "$run_name" != *_smoke ]]; then
  run_name="${run_name}_smoke"
fi
test -f "$fasta"
test -f "$targets"

EXTRA_ARGS=(--num-epochs "${NUM_EPOCHS:-100}")
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS=(--num-epochs 1 --max-train-steps 1 --limit-train 8 --limit-valid 8 --limit-test 8)
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  EXTRA_ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$run_name")
  [[ -n "${WANDB_ENTITY:-}" ]] && EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  EXTRA_ARGS+=(--resume-from "${RESUME_FROM}")
fi
if [[ "${REPORT_HEAD_GRADIENT_NORMS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--report-head-gradient-norms)
fi
if [[ "${EVALUATE_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--evaluate-only)
fi
if [[ -n "${CORRELATION_LOSS_WEIGHT:-}" ]]; then
  EXTRA_ARGS+=(
    --double-centered-correlation-loss-weight "${CORRELATION_LOSS_WEIGHT}"
  )
fi

"${HOME}/venv/jax/bin/python" scripts/run_humanbraindev_finetune.py \
  --targets-config "$targets" \
  --fasta-path "$fasta" \
  --organism "$organism" \
  --include-chroms "$include_chroms" \
  --valid-chroms "${VALID_CHROMS:-chr8}" \
  --test-chroms "${TEST_CHROMS:-chr9}" \
  --checkpoint-path "${CHECKPOINT_PATH:-/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1}" \
  --checkpoint-dir "checkpoints/v0data/${run_name}" \
  --pretrained-head-initialization "${PRETRAINED_HEAD_INITIALIZATION:-none}" \
  --split-source chromosome \
  --window-size 131072 \
  --stride 131072 \
  --batch-size "${BATCH_SIZE:-8}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --learning-rate-schedule "${LEARNING_RATE_SCHEDULE:-constant}" \
  --warmup-steps "${WARMUP_STEPS:-0}" \
  --minimum-learning-rate-ratio "${MINIMUM_LEARNING_RATE_RATIO:-0.1}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --backbone-lora \
  --adapter-strategy "$strategy" \
  --lora-rank "${LORA_RANK:-16}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --locon-rank "${LOCON_RANK:-4}" \
  --locon-alpha "${LOCON_ALPHA:-1}" \
  --base-param-dtype "${BASE_PARAM_DTYPE:-float32}" \
  --lora-param-dtype "${LORA_PARAM_DTYPE:-float32}" \
  --locon-param-dtype "${LOCON_PARAM_DTYPE:-float32}" \
  --activation-dtype "${ACTIVATION_DTYPE:-bfloat16}" \
  --backbone-compute-dtype "${BACKBONE_COMPUTE_DTYPE:-bfloat16}" \
  --base-compute-dtype "${BASE_COMPUTE_DTYPE:-bfloat16}" \
  --lora-compute-dtype "${LORA_COMPUTE_DTYPE:-bfloat16}" \
  --locon-compute-dtype "${LOCON_COMPUTE_DTYPE:-bfloat16}" \
  --best-metric valid/mean/differential_pearson_r \
  --best-metric-mode max \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-5}" \
  --eval-splits valid,test \
  --target-workers "${TARGET_WORKERS:-24}" \
  --window-workers "${WINDOW_WORKERS:-8}" \
  --prefetch-batches 2 \
  --progress-interval 100 \
  "${EXTRA_ARGS[@]}"
