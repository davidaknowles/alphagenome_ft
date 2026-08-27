#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data

sbatch_bin="${SBATCH_BIN:-sbatch}"
default_dataset_config="outputs/v0data/joint-objective-variants/metric-tempered/datasets.json"
dataset_config="$(realpath "${DATASET_CONFIG:-$default_dataset_config}")"
initializer="${PRETRAINED_HEAD_INITIALIZATION:-none}"
run_tag="${RUN_TAG:-}"
warmup_max_epochs="${WARMUP_MAX_EPOCHS:-20}"
warmup_patience="${WARMUP_PATIENCE:-5}"
balance_gene_windows="${BALANCE_GENE_WINDOWS:-0}"
if [[ ! "$run_tag" =~ ^[a-z0-9_]*$ ]]; then
  printf 'Invalid RUN_TAG, %s; use lowercase letters, numbers, and underscores.\n' \
    "$run_tag" >&2
  exit 2
fi
if [[ "$balance_gene_windows" != "0" && "$balance_gene_windows" != "1" ]]; then
  printf 'BALANCE_GENE_WINDOWS must be 0 or 1, got %s.\n' "$balance_gene_windows" >&2
  exit 2
fi
if [[ ! "$warmup_max_epochs" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$warmup_patience" =~ ^[1-9][0-9]*$ ]] ||
   (( warmup_patience > warmup_max_epochs )); then
  printf 'Warmup requires positive integer max epochs and patience <= max epochs.\n' >&2
  exit 2
fi
case "$initializer" in
  none) initializer_suffix="" ;;
  bootstrap|neural_bootstrap|neural_accessibility_bootstrap|semantic_neural_accessibility_bootstrap)
    initializer_suffix="_${initializer}"
    ;;
  *)
    printf 'Unsupported pretrained head initialization, %s\n' "$initializer" >&2
    exit 2
    ;;
esac
tag_suffix="${run_tag:+_${run_tag}}"
branch_tag="$run_tag"
if [[ "$initializer" != "none" ]]; then
  branch_tag="${branch_tag:+${branch_tag}_}${initializer}"
fi
run_suffix="_head_warmup_tempered${tag_suffix}${initializer_suffix}"
run_basename="joint_all_nonencode"
warmup_run="${run_basename}${run_suffix}"
exports="ALL,RUN_BASENAME=${run_basename},RUN_SUFFIX=${run_suffix},BACKBONE_LORA=0,PRETRAINED_HEAD_INITIALIZATION=${initializer},LEARNING_RATE=${WARMUP_LEARNING_RATE:-1e-3},NUM_EPOCHS=${warmup_max_epochs},EARLY_STOPPING_PATIENCE=${warmup_patience},BALANCE_GENE_WINDOWS=${balance_gene_windows},DATASET_CONFIG=${dataset_config},TARGET_WORKERS=${TARGET_WORKERS:-12},WINDOW_WORKERS=${WINDOW_WORKERS:-4}"

smoke_args=(--parsable --array=0 --time=00:30:00)
if [[ -n "${INITIAL_DEPENDENCY:-}" ]]; then
  smoke_args+=(--dependency="${INITIAL_DEPENDENCY}")
fi

smoke=$(
  "$sbatch_bin" "${smoke_args[@]}" \
    --export="${exports},SMOKE=1" scripts/v0data/slurm_joint_multidataset_adapters.sbatch
)
warmup=$(
  "$sbatch_bin" --parsable --array=0 --dependency="afterok:${smoke}_0" \
    --export="$exports" scripts/v0data/slurm_joint_multidataset_adapters.sbatch
)
branch=$(
  "$sbatch_bin" --parsable --dependency="afterok:${warmup}_0" \
    --export="ALL,SOURCE_RUN=checkpoints/v0data/${warmup_run},DATASET_CONFIG=${dataset_config},BRANCH_TAG=${branch_tag},BALANCE_GENE_WINDOWS=${balance_gene_windows},TARGET_WORKERS=${TARGET_WORKERS:-12},WINDOW_WORKERS=${WINDOW_WORKERS:-4}" \
    scripts/v0data/slurm_submit_joint_adapters_from_head_warmup.sbatch
)

printf 'head warmup initializer=%s smoke=%s full=%s adapter-branch=%s\n' \
  "$initializer" "$smoke" "$warmup" "$branch"
