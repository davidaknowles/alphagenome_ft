#!/bin/bash
#SBATCH --job-name=v0_branch_warm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v0data/v0_branch_warm-%j.out
#SBATCH --error=logs/v0data/v0_branch_warm-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome_fp4
export LD_LIBRARY_PATH="/nfs/sw/easybuild/software/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
source scripts/v0data/joint_continuation_lib.sh

sbatch_bin="${SBATCH_BIN:-sbatch}"
source_run="${SOURCE_RUN:?SOURCE_RUN is required}"
source_checkpoint="${source_run}/best"
source_epoch="$("${HOME}/venv/jax/bin/python" - "${source_checkpoint}/metrics.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    print(int(json.load(handle)["epoch"]))
PY
)"
dataset_config="${DATASET_CONFIG:?DATASET_CONFIG is required}"
branch_tag="${BRANCH_TAG:-}"
if [[ -n "$branch_tag" && ! "$branch_tag" =~ ^[A-Za-z0-9_]+$ ]]; then
  printf 'Invalid branch tag: %s\n' "$branch_tag" >&2
  exit 1
fi
tag_prefix="${branch_tag:+${branch_tag}_}"
snapshot="checkpoints/v0data/joint_all_nonencode_${tag_prefix}head_warmup_seed_epoch${source_epoch}"
learning_rate=3e-4
num_epochs="${NUM_EPOCHS:-100}"
target_workers="${TARGET_WORKERS:-12}"
window_workers="${WINDOW_WORKERS:-4}"
balance_gene_windows="${BALANCE_GENE_WINDOWS:-0}"
if [[ "$balance_gene_windows" != "0" && "$balance_gene_windows" != "1" ]]; then
  printf 'BALANCE_GENE_WINDOWS must be 0 or 1, got %s.\n' "$balance_gene_windows" >&2
  exit 2
fi
expand_backbone_adapters=1
lora_rank=16
lora_alpha=16
locon_rank=4
locon_alpha=1
locon_targets="${LOCON_TARGETS:-downres_block_2;downres_block_3;downres_block_4;downres_block_5}"
GPU_GRES="${GPU_GRES:-gpu:l40s:2}"

snapshot_checkpoint "$source_checkpoint" "$source_epoch" "$snapshot"
test -f "$dataset_config"

suffix="_${tag_prefix}headwarm${source_epoch}_adapters_lr3e4_reset"
submit_continuation 0 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  "$suffix" "$dataset_config"
submit_continuation 1 "$source_run" "$source_epoch" "$snapshot" "$learning_rate" \
  "$suffix" "$dataset_config"
