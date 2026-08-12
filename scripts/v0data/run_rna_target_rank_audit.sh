#!/bin/bash
# Reproduce the gene-level RNA target-rank audit.

set -euo pipefail

cd "$(dirname "$0")/../.."
"${HOME}/venv/jax/bin/python" scripts/v0data/audit_rna_target_rank.py \
  --dataset HDA=outputs/v0data/hda-joint/gene_expression_supervision.npz \
  --dataset Liu=outputs/v0data/liu-hdma/joint/gene_expression_supervision.npz \
  --dataset Johansen-human=outputs/v0data/johansen-rna-corrected/supervision/human/gene_expression_supervision.npz \
  --dataset Johansen-macaque=outputs/v0data/johansen-rna-corrected/supervision/macaque/gene_expression_supervision.npz \
  --dataset Johansen-marmoset=outputs/v0data/johansen-rna-corrected/supervision/marmoset/gene_expression_supervision.npz \
  --json-output results/v0data_rna_target_rank.json \
  --markdown-output results/v0data_rna_target_rank.md
