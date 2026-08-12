#!/bin/bash
# Reproduce the cross-study gene-correlation centering audit.

set -euo pipefail

cd "$(dirname "$0")/../.."
human_fai=/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa.fai
macaque_include='NC_041754.1;NC_041755.1;NC_041756.1;NC_041757.1;NC_041758.1;NC_041759.1;NC_041760.1;NC_041761.1;NC_041762.1;NC_041763.1;NC_041764.1;NC_041765.1;NC_041766.1;NC_041767.1;NC_041768.1;NC_041769.1;NC_041770.1;NC_041771.1;NC_041772.1;NC_041773.1;NC_041774.1'

"${HOME}/venv/jax/bin/python" scripts/v0data/audit_gene_correlation_centering.py \
  --dataset HDA=outputs/v0data/hda-joint/gene_expression_supervision.npz \
  --dataset Liu=outputs/v0data/liu-hdma/joint/gene_expression_supervision.npz \
  --dataset Johansen-human=outputs/v0data/johansen-rna-corrected/supervision/human/gene_expression_supervision.npz \
  --dataset Johansen-macaque=outputs/v0data/johansen-rna-corrected/supervision/macaque/gene_expression_supervision.npz \
  --dataset Johansen-marmoset=outputs/v0data/johansen-rna-corrected/supervision/marmoset/gene_expression_supervision.npz \
  --dataset Zemke2023-published-human=outputs/v0data/zemke2023-published-gene-supervision/human/gene_expression_supervision.npz \
  --fasta-index HDA="$human_fai" \
  --fasta-index Liu="$human_fai" \
  --fasta-index Johansen-human="$human_fai" \
  --fasta-index Johansen-macaque=outputs/allen_brain_multiome_multispecies_v1/references/macaque.fa.fai \
  --fasta-index Johansen-marmoset=outputs/allen_brain_multiome_multispecies_v1/references/marmoset.fa.fai \
  --fasta-index Zemke2023-published-human="$human_fai" \
  --held-out Johansen-macaque=NC_041761.1,NC_041762.1 \
  --include-chromosomes "Johansen-macaque=$macaque_include" \
  --dataset-batch-size Johansen-human=4 \
  --dataset-batch-size Johansen-macaque=4 \
  --dataset-batch-size Johansen-marmoset=4 \
  --json-output results/v0data_gene_correlation_centering.json \
  --markdown-output results/v0data_gene_correlation_centering.md
