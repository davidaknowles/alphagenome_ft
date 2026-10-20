#!/bin/bash
# Collate eQTL reruns: per-dataset scores, distance plots, overall and
# within-panel performance, gene-span distance figure and track specificity.
#SBATCH --job-name=eqtl_collate
#SBATCH --partition=cpu,io
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eqtl/collate-%j.out
#SBATCH --error=logs/eqtl/collate-%j.err

set -euo pipefail
cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
python="${HOME}/venv/jax/bin/python"
root="${EQTL_OUTPUT_ROOT:?EQTL_OUTPUT_ROOT is required}"
declare -A native=([singlebrain]="results/eqtl/native_1mb_all_tracks/specificity/native_biological_track_scores.tsv" [xqtl]="results/eqtl_xqtl/native_analysis/native_biological_track_scores.tsv")
for dataset in singlebrain xqtl; do
  "$python" scripts/eqtl/collate.py "${root}/${dataset}"
  "$python" scripts/eqtl/plot.py "${root}/${dataset}/scores.tsv" --output-dir "${root}/plots_${dataset}" --native-scores "${native[$dataset]}"
done
"$python" scripts/eqtl/summarize_performance.py \
  --dataset "singlebrain=${root}/singlebrain/scores.tsv=${native[singlebrain]}" \
  --dataset "xqtl=${root}/xqtl/scores.tsv=${native[xqtl]}" \
  --output "${root}/overall_performance.tsv"
"$python" scripts/eqtl/plot_gene_span_by_distance.py \
  --summary "SingleBrain=${root}/plots_singlebrain/summary.tsv" --summary "XQTL=${root}/plots_xqtl/summary.tsv" \
  --output "${root}/gene_span_auroc_by_distance"
"$python" scripts/eqtl/analyze_trained_specificity.py \
  --dataset "singlebrain=${root}/singlebrain" --dataset "xqtl=${root}/xqtl" --output-dir "${root}/specificity"
