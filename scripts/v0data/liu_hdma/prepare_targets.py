#!/usr/bin/env python
"""Combine reconstructed Liu HDMA ATAC and exon-plus-gene RNA targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import (
    build_fasta_index,
    read_gene_exons,
    read_pseudobulk_expression,
    write_gene_expression_supervision,
    write_stranded_exon_bigwigs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-pseudobulk", required=True, type=Path)
    parser.add_argument("--atac-targets", required=True, type=Path)
    parser.add_argument("--gtf", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression = read_pseudobulk_expression(args.rna_pseudobulk, normalize_cpm=False)
    atac_payload = json.loads(args.atac_targets.read_text())
    if len(atac_payload.get("heads", ())) != 1:
        raise ValueError("Expected one reconstructed ATAC head.")
    atac_head = atac_payload["heads"][0]
    atac_groups = tuple(target["label"] for target in atac_head["targets"])
    if atac_groups != expression.groups:
        raise ValueError("ATAC and RNA cluster order differs.")

    chromosome_sizes = build_fasta_index(args.fasta.expanduser().resolve())
    genes = read_gene_exons(
        args.gtf.expanduser().resolve(),
        gene_ids=expression.gene_ids,
        chromosome_sizes=chromosome_sizes,
    )
    coverage = len(genes) / len(expression.gene_ids)
    if coverage < 0.9:
        raise ValueError(
            f"Only {len(genes)}/{len(expression.gene_ids)} RNA genes ({coverage:.1%}) matched."
        )

    output_dir = args.output_dir.expanduser().resolve()
    rna_targets = write_stranded_exon_bigwigs(
        expression,
        genes=genes,
        chromosome_sizes=chromosome_sizes,
        output_dir=output_dir / "rna_exon_cpm",
        overwrite=args.overwrite,
    )
    supervision_path = output_dir / "gene_expression_supervision.npz"
    matched_genes = write_gene_expression_supervision(
        supervision_path,
        expression,
        genes=genes,
    )
    atac_head["id"] = "liu_atac"
    config = {
        "dataset": "liu_hdma",
        "heads": [
            atac_head,
            {
                "id": "liu_rna",
                "source": "predefined",
                "kind": "rna_seq",
                "resolutions": [1, 128],
                "apply_squashing": True,
                "gene_supervision": {
                    "path": str(supervision_path),
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
        "clusters": len(expression.groups),
        "atac_tracks": len(atac_groups),
        "rna_tracks": len(rna_targets),
        "input_genes": len(expression.gene_ids),
        "matched_genes": matched_genes,
        "rna_source": "raw UMI counts aggregated by published cell cluster and normalized to CPM",
        "rna_representation": "strand-specific union-exon density with gene-level supervision",
        "atac_source": "all selected fragments normalized as coverage per million fragments",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
