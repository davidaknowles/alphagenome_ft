#!/bin/bash
#SBATCH -J johansen_atac_chrom
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH --array=0-43%8
#SBATCH -o logs/allen_atac_reprocess/%x-%A_%a.out
#SBATCH -e logs/allen_atac_reprocess/%x-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/allen_atac_reprocess outputs/v0data/johansen-fragment-atac
module load HTSlib/1.24-GCC-14.2.0
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

species_config=outputs/allen_brain_multiome_multispecies_v1/species.json
mapfile -t jobs < <(
  python scripts/allen_atac_reprocessing/species_config_query.py \
    --species-config "$species_config" \
    chromosome-jobs \
    --species macaque marmoset
)
if [[ "${#jobs[@]}" -ne 44 ]]; then
  echo "Expected 44 nonhuman primary-chromosome jobs, found ${#jobs[@]}." >&2
  exit 2
fi
IFS=$'\t' read -r species chromosome fasta <<< "${jobs[$SLURM_ARRAY_TASK_ID]}"
case "$species" in
  macaque) title=Macaque ;;
  marmoset) title=Marmoset ;;
  *) echo "Unsupported species $species" >&2; exit 2 ;;
esac

root=/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/johansen2025Crossspecies
metadata="$root/comprehensive_expression_matrix/$title/${title}_HMBA_basalganglia_AIT_pre-print.h5ad"
chromosome_size="$(awk -v chromosome="$chromosome" '$1 == chromosome {print $2}' "${fasta}.fai")"
test -n "$chromosome_size"
extra_args=()
if [[ "$species" == "macaque" ]]; then
  extra_args+=(--fragment-cell-id-mode library_barcode)
elif [[ "$species" == "marmoset" ]]; then
  extra_args+=(--fragment-library-suffix .h5ad)
fi

python scripts/allen_atac_reprocessing/aggregate.py \
  --fragment-root "$root/fragment/$species" \
  --cell-metadata-h5ad "$metadata" \
  --chromosome "$chromosome" \
  --chromosome-size "$chromosome_size" \
  --output "outputs/v0data/johansen-fragment-atac/$species/shards/${chromosome}_100bp.npz" \
  "${extra_args[@]}"
