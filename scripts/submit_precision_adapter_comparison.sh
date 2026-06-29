#!/bin/bash
set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/slurm

OUTPUT_ROOT="${1:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4/outputs/precision_adapter_compare/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_ROOT}"

for backend in jax torch; do
  for precision in default nvfp8; do
    for strategy in lora lora+locon; do
      job_name="ag-${backend}-${precision}-${strategy//+/-}"
      sbatch \
        -J "${job_name}" \
        --export=ALL,AG_BACKEND="${backend}",AG_PRECISION="${precision}",AG_ADAPTER_STRATEGY="${strategy}",AG_OUTPUT_ROOT="${OUTPUT_ROOT}" \
        scripts/slurm_precision_adapter_cell.sbatch
    done
  done
done

echo "${OUTPUT_ROOT}"
