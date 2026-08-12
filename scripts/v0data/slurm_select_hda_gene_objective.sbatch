#!/bin/bash
#SBATCH --job-name=hda_objective_select
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/v0data/hda_objective_select-%j.out
#SBATCH --error=logs/v0data/hda_objective_select-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
scripts/v0data/select_and_submit_hda_gene_objective_match.sh
