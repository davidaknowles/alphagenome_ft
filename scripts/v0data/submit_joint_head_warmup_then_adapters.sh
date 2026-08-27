#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data

sbatch_bin="${SBATCH_BIN:-sbatch}"
dataset_config="$(realpath outputs/v0data/joint-objective-variants/metric-tempered/datasets.json)"
initializer="${PRETRAINED_HEAD_INITIALIZATION:-none}"
case "$initializer" in
  none) initializer_suffix=""; branch_tag="" ;;
  bootstrap|neural_bootstrap|neural_accessibility_bootstrap)
    initializer_suffix="_${initializer}"
    branch_tag="$initializer"
    ;;
  *)
    printf 'Unsupported pretrained head initialization, %s\n' "$initializer" >&2
    exit 2
    ;;
esac
run_suffix="_head_warmup_tempered${initializer_suffix}"
run_basename="joint_all_nonencode"
warmup_run="${run_basename}${run_suffix}"
exports="ALL,RUN_BASENAME=${run_basename},RUN_SUFFIX=${run_suffix},BACKBONE_LORA=0,PRETRAINED_HEAD_INITIALIZATION=${initializer},LEARNING_RATE=${WARMUP_LEARNING_RATE:-1e-3},NUM_EPOCHS=${WARMUP_EPOCHS:-3},EARLY_STOPPING_PATIENCE=${WARMUP_EPOCHS:-3},DATASET_CONFIG=${dataset_config},TARGET_WORKERS=${TARGET_WORKERS:-12},WINDOW_WORKERS=${WINDOW_WORKERS:-4}"

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
    --export="ALL,SOURCE_RUN=checkpoints/v0data/${warmup_run},DATASET_CONFIG=${dataset_config},BRANCH_TAG=${branch_tag},TARGET_WORKERS=${TARGET_WORKERS:-12},WINDOW_WORKERS=${WINDOW_WORKERS:-4}" \
    scripts/v0data/slurm_submit_joint_adapters_from_head_warmup.sbatch
)

printf 'head warmup initializer=%s smoke=%s full=%s adapter-branch=%s\n' \
  "$initializer" "$smoke" "$warmup" "$branch"
