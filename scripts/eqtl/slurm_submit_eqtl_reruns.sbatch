#!/bin/bash
# Queue SingleBrain and XQTL benchmarks for a head-only snapshot and the best
# LoRA and LoRA+LoCon checkpoints, then a collation job. Submitted by the
# adapter branch dispatcher through POST_ADAPTER_SBATCH.
#SBATCH --job-name=eqtl_dispatch
#SBATCH --partition=cpu,io
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/eqtl/dispatch-%j.out
#SBATCH --error=logs/eqtl/dispatch-%j.err

set -euo pipefail
cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
: "${HEAD_ONLY_CHECKPOINT:?}" "${LORA_RUN:?}" "${LOCON_RUN:?}" "${DATASET_CONFIG:?}" "${EQTL_OUTPUT_ROOT:?}"
base="${EQTL_BASE_CHECKPOINT:-${CHECKPOINT_PATH:?CHECKPOINT_PATH or EQTL_BASE_CHECKPOINT is required}}"
declare -A models=([head_only]="$HEAD_ONLY_CHECKPOINT" [lora]="${LORA_RUN}/best" [locon]="${LOCON_RUN}/best")
declare -A records=([singlebrain]="results/eqtl/finemap_1mb/records.jsonl" [xqtl]="results/eqtl_xqtl/prepared/records.jsonl")
for path in "${models[@]}"; do test -f "${path}/config.json"; done
jobs=()
for dataset in singlebrain xqtl; do
  for label in head_only lora locon; do
    job=$(sbatch --parsable -J "eqtl-${dataset}-${label}" \
      --export="ALL,MODEL_ARG=${label}=${models[$label]},OUTPUT_DIR=${EQTL_OUTPUT_ROOT}/${dataset},RECORDS_CACHE=${records[$dataset]},BASE_CHECKPOINT=${base},DATASET_CONFIG=${DATASET_CONFIG}" \
      scripts/eqtl/run_eqtl_benchmark.sbatch)
    jobs+=("$job")
    printf '%s %s %s\n' "$dataset" "$label" "$job"
  done
done
collate=$(sbatch --parsable --dependency="afterok:$(IFS=:; echo "${jobs[*]}")" \
  --export="ALL,EQTL_OUTPUT_ROOT=${EQTL_OUTPUT_ROOT}" scripts/eqtl/slurm_collate_eqtl_reruns.sbatch)
printf 'collate %s\n' "$collate"
