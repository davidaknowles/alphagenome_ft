#!/bin/bash
set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/v0data outputs/v0data/liu-hdma
root=/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/liu2026Multiomics

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
"${HOME}/venv/jax/bin/python" scripts/v0data/liu_hdma/audit_cells.py \
  --metadata outputs/v0data/liu2026/source/per_cell_meta.csv \
  --supplementary-table "$root/supplementary_table/S2/S2.xlsx" \
  --bigwig-root "$root/bigwig" \
  --expression-root "$root/sample_cell_expression_matrix" \
  --output outputs/v0data/liu-hdma/clusters.json

rna_samples="$(sbatch --parsable scripts/v0data/liu_hdma/slurm_rna_samples.sbatch)"
rna_reduce="$(sbatch --parsable --dependency="afterok:${rna_samples}" scripts/v0data/liu_hdma/slurm_rna_reduce.sbatch)"
atac_totals="$(sbatch --parsable scripts/v0data/liu_hdma/slurm_atac_totals.sbatch)"
atac_chromosomes="$(sbatch --parsable --dependency="afterok:${atac_totals}" scripts/v0data/liu_hdma/slurm_atac_chromosomes.sbatch)"
atac_bigwigs="$(sbatch --parsable --dependency="afterok:${atac_chromosomes}" scripts/v0data/liu_hdma/slurm_materialize_atac.sbatch)"
targets="$(sbatch --parsable --dependency="afterok:${rna_reduce}:${atac_bigwigs}" scripts/v0data/liu_hdma/slurm_prepare_targets.sbatch)"

printf 'RNA samples %s\nRNA reduce %s\nATAC totals %s\nATAC chromosomes %s\nATAC BigWigs %s\nJoint targets %s\n' \
  "$rna_samples" "$rna_reduce" "$atac_totals" "$atac_chromosomes" "$atac_bigwigs" "$targets"
