#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data evaluations/v0data/joint_all_nonencode

strategy="${STRATEGY:?Set STRATEGY to lora or lora_locon}"
source_checkpoint="${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT to a selected checkpoint}"
evaluation_tag="${EVALUATION_TAG:?Set EVALUATION_TAG to a unique result name}"
source_job="${SOURCE_JOB:-}"
dataset_config="${DATASET_CONFIG:-outputs/v0data/joint-all-nonencode/datasets.json}"
sbatch_bin="${SBATCH_BIN:-sbatch}"
gpu_gres="${GPU_GRES:-gpu:l40s:2}"
locon_targets="${LOCON_TARGETS:-default}"

if [[ "$strategy" != lora && "$strategy" != lora_locon ]]; then
  printf 'Invalid strategy: %s\n' "$strategy" >&2
  exit 1
fi
if [[ ! "$evaluation_tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'Evaluation tag contains unsupported characters: %s\n' "$evaluation_tag" >&2
  exit 1
fi
test -f "$dataset_config"
test -f "$source_checkpoint/metrics.json"

sbatch_args=(
  --parsable
  --array=0-9%4
  --gres="$gpu_gres"
  --export="ALL,EVALUATION_STRATEGY=${strategy},SOURCE_CHECKPOINT=${source_checkpoint},EVALUATION_TAG=${evaluation_tag},DATASET_CONFIG=${dataset_config},LOCON_TARGETS=${locon_targets},TARGET_CACHE_DIR=${TARGET_CACHE_DIR:-},TARGET_CACHE_SPLITS=${TARGET_CACHE_SPLITS:-valid;test},TARGET_CACHE_DTYPE=${TARGET_CACHE_DTYPE:-float16}"
)
if [[ -n "$source_job" ]]; then
  sbatch_args+=(--dependency="afterok:${source_job}")
fi
job=$("$sbatch_bin" "${sbatch_args[@]}" scripts/v0data/slurm_joint_multidataset_evaluate.sbatch)
printf '%s native-source evaluations=%s\n' "$evaluation_tag" "$job"
