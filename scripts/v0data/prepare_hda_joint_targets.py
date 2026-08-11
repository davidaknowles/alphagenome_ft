#!/usr/bin/env python
"""Prepare paired ATAC and exon-plus-gene RNA targets for the Mannens atlas."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import (
    PseudobulkExpression,
    build_fasta_index,
    read_gene_exons,
    read_pseudobulk_expression,
    write_gene_expression_supervision,
    write_stranded_exon_bigwigs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5ad", required=True, type=Path)
    parser.add_argument("--atac-bigwig-dir", required=True, type=Path)
    parser.add_argument("--gtf", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def main() -> None:
    args = _parse_args()
    expression = read_pseudobulk_expression(
        args.rna_h5ad,
        normalize_cpm=False,
        gene_id_column="Accession",
        group_column=None,
        matrix_key="layers/CPM",
    )
    atac_by_group = {
        path.stem: path.resolve() for path in args.atac_bigwig_dir.expanduser().glob("*.bw")
    }
    expression_index = {group: index for index, group in enumerate(expression.groups)}
    paired_groups = tuple(
        group for group in expression.groups if _safe_name(group) in atac_by_group
    )
    omitted_rna_groups = sorted(set(expression.groups) - set(paired_groups))
    extra_atac_groups = sorted(set(atac_by_group) - {_safe_name(group) for group in paired_groups})
    if extra_atac_groups:
        raise ValueError(f"ATAC groups without RNA expression: {extra_atac_groups}")
    expression = PseudobulkExpression(
        groups=paired_groups,
        gene_ids=expression.gene_ids,
        cpm=np.stack([expression.cpm[expression_index[group]] for group in paired_groups]),
    )

    chromosome_sizes = build_fasta_index(args.fasta.expanduser().resolve())
    genes = read_gene_exons(
        args.gtf.expanduser().resolve(),
        gene_ids=expression.gene_ids,
        chromosome_sizes=chromosome_sizes,
    )
    coverage = len(genes) / len(expression.gene_ids)
    if coverage < 0.9:
        raise ValueError(
            f"Only {len(genes)}/{len(expression.gene_ids)} RNA genes ({coverage:.1%}) "
            "matched the GTF and FASTA."
        )

    output_dir = args.output_dir.expanduser().resolve()
    rna_targets = write_stranded_exon_bigwigs(
        expression,
        genes=genes,
        chromosome_sizes=chromosome_sizes,
        output_dir=output_dir / "rna_exon_cpm",
        overwrite=args.overwrite,
    )
    gene_supervision_path = output_dir / "gene_expression_supervision.npz"
    matched_genes = write_gene_expression_supervision(
        gene_supervision_path,
        expression,
        genes=genes,
    )
    atac_targets = [
        {
            "path": str(atac_by_group[_safe_name(group)]),
            "label": group,
            "strand": ".",
        }
        for group in paired_groups
    ]
    config = {
        "dataset": "hda-joint",
        "heads": [
            {
                "id": "hda_atac",
                "source": "predefined",
                "kind": "atac",
                "resolutions": [1, 128],
                "apply_squashing": False,
                "targets": atac_targets,
            },
            {
                "id": "hda_rna",
                "source": "predefined",
                "kind": "rna_seq",
                "resolutions": [1, 128],
                "apply_squashing": True,
                "gene_supervision": {
                    "path": str(gene_supervision_path),
                    "loss_weight": 1.0,
                    "coverage_loss_weight": 0.1,
                },
                "targets": rna_targets,
            },
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "targets.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "paired_groups": len(paired_groups),
        "omitted_rna_groups": omitted_rna_groups,
        "input_genes": len(expression.gene_ids),
        "matched_genes": matched_genes,
        "atac_tracks": len(atac_targets),
        "rna_tracks": len(rna_targets),
        "rna_source": "published CPM layer",
        "rna_representation": "strand-specific union-exon density with gene-level supervision",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
