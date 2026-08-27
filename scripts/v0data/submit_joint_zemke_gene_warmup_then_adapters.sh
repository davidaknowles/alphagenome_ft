#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data checkpoints/v0data

sbatch_bin="${SBATCH_BIN:-sbatch}"
species_dir="outputs/v0data/zemke2023-gene-only-species"
base_dir="outputs/v0data/joint-all-nonencode-zemke-gene"
config_dir="outputs/v0data/joint-objective-variants/metric-tempered-zemke-gene"
warmup_max_epochs="${WARMUP_MAX_EPOCHS:-20}"
warmup_patience="${WARMUP_PATIENCE:-5}"
locon_targets="${LOCON_TARGETS:-downres_block_2;downres_block_3;downres_block_4;downres_block_5}"
if [[ ! "$warmup_max_epochs" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$warmup_patience" =~ ^[1-9][0-9]*$ ]] ||
   (( warmup_patience > warmup_max_epochs )); then
  printf 'Warmup requires positive integer max epochs and patience <= max epochs.\n' >&2
  exit 2
fi

"${HOME}/venv/jax/bin/python" scripts/v0data/zemke2023_rna_reprocessing/prepare_gene_only_species.py \
  --input outputs/v0data/zemke2023-species/species.json \
  --supervision-root outputs/v0data/zemke2023-gene-supervision \
  --output-dir "$species_dir" \
  --correlation-loss-weight 1
"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_joint_dataset_config.py \
  --zemke2023-species "${species_dir}/species.json" \
  --output "${base_dir}/datasets.json"
"${HOME}/venv/jax/bin/python" scripts/v0data/prepare_joint_metric_aligned_config.py \
  --input "${base_dir}/datasets.json" \
  --output-dir "$config_dir" \
  --zemke-weight 3 \
  --zemke-rna-weight 1

dataset_config="$(realpath "${config_dir}/datasets.json")"
run_suffix="_head_warmup_tempered_zemke_gene"
warmup_run="joint_all_nonencode${run_suffix}"
exports="ALL,RUN_BASENAME=joint_all_nonencode,RUN_SUFFIX=${run_suffix},BACKBONE_LORA=0,LEARNING_RATE=${WARMUP_LEARNING_RATE:-1e-3},NUM_EPOCHS=${warmup_max_epochs},EARLY_STOPPING_PATIENCE=${warmup_patience},DATASET_CONFIG=${dataset_config},TARGET_WORKERS=${TARGET_WORKERS:-12},WINDOW_WORKERS=${WINDOW_WORKERS:-4},SMOKE_LIMIT_TRAIN=${SMOKE_LIMIT_TRAIN:-40},SMOKE_LIMIT_VALID=${SMOKE_LIMIT_VALID:-40},SMOKE_LIMIT_TEST=${SMOKE_LIMIT_TEST:-40}"

smoke=$(
  "$sbatch_bin" --parsable --array=0 --time=00:30:00 \
    --export="${exports},SMOKE=1" scripts/v0data/slurm_joint_multidataset_adapters.sbatch
)
warmup=$(
  "$sbatch_bin" --parsable --array=0 --dependency="afterok:${smoke}_0" \
    --export="$exports" scripts/v0data/slurm_joint_multidataset_adapters.sbatch
)
branch=$(
  "$sbatch_bin" --parsable --dependency="afterok:${warmup}_0" \
    --export="ALL,SOURCE_RUN=checkpoints/v0data/${warmup_run},DATASET_CONFIG=${dataset_config},BRANCH_TAG=zemke_gene,LOCON_TARGETS=${locon_targets},TARGET_WORKERS=${TARGET_WORKERS:-12},WINDOW_WORKERS=${WINDOW_WORKERS:-4}" \
    scripts/v0data/slurm_submit_joint_adapters_from_head_warmup.sbatch
)

printf 'Zemke direct-gene head warmup smoke=%s full=%s adapter-branch=%s\n' \
  "$smoke" "$warmup" "$branch"
