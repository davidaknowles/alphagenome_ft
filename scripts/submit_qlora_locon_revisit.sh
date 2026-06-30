#!/bin/bash
set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/slurm

OUTPUT_ROOT="${1:-/gpfs/commons/home/daknowles/projects/alphagenome_fp4/outputs/qlora_locon_revisit/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_ROOT}"

# Revisit only LoRA+LoCon with 16-bit trainable params.  The bf16 cell is the
# 16-bit non-quantized baseline; nvfp8/nvfp4 are the QLoRA-style base-quantized
# cells suggested by the quant-ablation results.
for backend in jax torch; do
  for precision in bf16 nvfp8 nvfp4; do
    job_name="ag-${backend}-${precision}-locon16"
    sbatch \
      -J "${job_name}" \
      --export=ALL,AG_BACKEND="${backend}",AG_PRECISION="${precision}",AG_ADAPTER_STRATEGY="lora+locon",AG_OUTPUT_ROOT="${OUTPUT_ROOT}" \
      scripts/slurm_precision_adapter_cell.sbatch
  done
done

echo "${OUTPUT_ROOT}"
