#!/bin/bash
# Submit matched one-epoch pretrained-head bootstrap screens.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/v0data

initialization=${PRETRAINED_HEAD_INITIALIZATION:-bootstrap}
case "$initialization" in
  bootstrap) run_suffix=_bootstrap_screen ;;
  neural_bootstrap) run_suffix=_neural_bootstrap_screen ;;
  *)
    echo "Unsupported pretrained-head initialization: $initialization" >&2
    exit 2
    ;;
esac

submit_screen() {
  local dataset=$1
  local targets=$2
  sbatch --parsable --array=0-1%2 \
    --export="ALL,DATASET=${dataset},TARGETS_CONFIG=${targets},RUN_SUFFIX=${run_suffix},NUM_EPOCHS=1,PRETRAINED_HEAD_INITIALIZATION=${initialization}" \
    scripts/v0data/slurm_joint_adapter_comparison.sbatch
}

printf 'hda=%s\n' "$(submit_screen hda-joint outputs/v0data/hda-joint/targets.json)"
printf 'liu=%s\n' "$(submit_screen liu-hdma outputs/v0data/liu-hdma/joint/targets_geneonly_corrw1.json)"
