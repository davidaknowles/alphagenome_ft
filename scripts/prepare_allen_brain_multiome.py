#!/usr/bin/env python
"""Prepare paired ATAC and RNA pseudobulk targets for Allen Brain Multiome."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import (
    build_fasta_index,
    read_gene_bodies,
    read_pseudobulk_expression,
    write_stranded_gene_body_bigwigs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5ad", type=Path, required=True)
    parser.add_argument("--atac-bigwig-dir", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    rna_dir = output_dir / "rna_gene_body_cpm"
    expression = read_pseudobulk_expression(args.rna_h5ad, normalize_cpm=True)
    chromosome_sizes = build_fasta_index(args.fasta.expanduser().resolve())
    gene_bodies = read_gene_bodies(
        args.gtf.expanduser().resolve(),
        gene_ids=expression.gene_ids,
        chromosome_sizes=chromosome_sizes,
    )
    coverage = len(gene_bodies) / len(expression.gene_ids)
    if coverage < 0.9:
        raise ValueError(
            f"Only {len(gene_bodies)}/{len(expression.gene_ids)} RNA genes ({coverage:.1%}) "
            "matched the GTF and FASTA chromosomes."
        )

    atac_dir = args.atac_bigwig_dir.expanduser().resolve()
    atac_paths = sorted(atac_dir.glob("*.bw"))
    atac_by_group = {path.stem: path for path in atac_paths}
    missing_atac = [group for group in expression.groups if safe_name(group) not in atac_by_group]
    extra_atac = sorted(set(atac_by_group) - {safe_name(group) for group in expression.groups})
    if missing_atac or extra_atac:
        raise ValueError(
            f"ATAC/RNA group mismatch, missing ATAC={missing_atac}, extra ATAC={extra_atac}."
        )

    rna_targets = write_stranded_gene_body_bigwigs(
        expression,
        gene_bodies=gene_bodies,
        chromosome_sizes=chromosome_sizes,
        output_dir=rna_dir,
        overwrite=args.overwrite,
    )
    atac_targets = [
        {
            "path": str(atac_by_group[safe_name(group)]),
            "label": group,
            "strand": ".",
        }
        for group in expression.groups
    ]
    config = {
        "heads": [
            {
                "id": "allen_atac",
                "source": "predefined",
                "kind": "atac",
                "resolutions": [1, 128],
                "apply_squashing": False,
                "targets": atac_targets,
            },
            {
                "id": "allen_rna",
                "source": "predefined",
                "kind": "rna_seq",
                "resolutions": [1, 128],
                "apply_squashing": True,
                "targets": rna_targets,
            },
        ]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "targets.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "rna_h5ad": str(args.rna_h5ad.expanduser().resolve()),
        "atac_bigwig_dir": str(atac_dir),
        "gtf": str(args.gtf.expanduser().resolve()),
        "fasta": str(args.fasta.expanduser().resolve()),
        "groups": len(expression.groups),
        "input_genes": len(expression.gene_ids),
        "matched_genes": len(gene_bodies),
        "atac_tracks": len(atac_targets),
        "rna_tracks": len(rna_targets),
        "rna_representation": "strand-specific gene-body density with each gene integral equal to CPM",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"Targets config written to {config_path}")


if __name__ == "__main__":
    main()
