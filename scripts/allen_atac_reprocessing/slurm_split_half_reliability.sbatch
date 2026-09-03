#!/bin/bash
#SBATCH --job-name=johansen_atac_reliability
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/allen_atac_reprocess/johansen_atac_reliability-%A_%a.out
#SBATCH --error=logs/allen_atac_reprocess/johansen_atac_reliability-%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
mkdir -p logs/allen_atac_reprocess results
module load HTSlib/1.24-GCC-14.2.0
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source ~/venv/jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

root=/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary/johansen2025Crossspecies
case "$SLURM_ARRAY_TASK_ID" in
  0)
    species=human
    chromosome=chr9
    fasta=/gpfs/commons/groups/knowles_lab/index/hg38/hg38.fa
    title=Human
    extra_args=()
    ;;
  1)
    species=macaque
    chromosome=NC_041762.1
    fasta=outputs/allen_brain_multiome_multispecies_v1/references/macaque.fa
    title=Macaque
    extra_args=(--fragment-cell-id-mode library_barcode)
    ;;
  2)
    species=marmoset
    chromosome=chr9
    fasta=outputs/allen_brain_multiome_multispecies_v1/references/marmoset.fa
    title=Marmoset
    extra_args=(--fragment-library-suffix .h5ad)
    ;;
  *)
    echo "Unknown array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac
chromosome_size="$(awk -v chromosome="$chromosome" '$1 == chromosome {print $2}' "${fasta}.fai")"
test -n "$chromosome_size"

python scripts/allen_atac_reprocessing/audit_split_half.py \
  --fragment-root "$root/fragment/$species" \
  --cell-metadata-h5ad "$root/comprehensive_expression_matrix/$title/${title}_HMBA_basalganglia_AIT_pre-print.h5ad" \
  --chromosome "$chromosome" \
  --chromosome-size "$chromosome_size" \
  --output "results/v0data_johansen_${species}_atac_split_half_reliability.json" \
  "${extra_args[@]}"
