#!/usr/bin/env python3
"""Add raw-count gene supervision to one Zemke 2023 target manifest."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import (
    PseudobulkExpression,
    build_fasta_index,
    read_gene_exons,
    write_gene_expression_supervision,
)
from alphagenome_ft.finetune.reprocessing import (
    aggregate_matrix_market_by_group,
    normalize_counts_per_million,
    read_10x_features,
)


def safe_group_name(value: str) -> str:
    """Normalize release subclass labels to published-track labels."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def read_barcode_groups(
    metadata_path: Path,
    *,
    target_groups: tuple[str, ...],
    cell_column: str = "cell",
    group_column: str = "subclass",
) -> tuple[dict[str, str], dict[str, int]]:
    """Read cell labels and retain only groups represented by published tracks."""
    normalized_targets = {safe_group_name(group): group for group in target_groups}
    if len(normalized_targets) != len(target_groups):
        raise ValueError("Target group labels are not unique after normalization.")
    barcode_groups: dict[str, str] = {}
    excluded: dict[str, int] = {}
    with _open_text(metadata_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {cell_column, group_column} <= set(reader.fieldnames):
            raise ValueError(
                f"{metadata_path} must contain {cell_column!r} and {group_column!r}."
            )
        for row in reader:
            barcode = row[cell_column]
            source_group = row[group_column]
            target_group = normalized_targets.get(safe_group_name(source_group))
            if target_group is None:
                excluded[source_group] = excluded.get(source_group, 0) + 1
                continue
            if barcode in barcode_groups:
                raise ValueError(f"Duplicate metadata barcode {barcode!r}.")
            barcode_groups[barcode] = target_group
    counts = {group: 0 for group in target_groups}
    for group in barcode_groups.values():
        counts[group] += 1
    missing = [group for group, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Metadata has no cells for published groups: {missing}.")
    return barcode_groups, excluded


def _rna_head(config: dict[str, Any]) -> dict[str, Any]:
    heads = [
        head
        for head in config.get("heads", ())
        if str(head.get("kind", "")).lower() == "rna_seq"
    ]
    if len(heads) != 1:
        raise ValueError(f"Expected one RNA-seq head, found {len(heads)}.")
    return heads[0]


def prepare_gene_supervision(
    *,
    matrix_path: Path,
    barcode_path: Path,
    feature_path: Path,
    metadata_path: Path,
    targets_path: Path,
    gtf_path: Path,
    fasta_path: Path,
    output_dir: Path,
    species: str,
    minimum_gene_coverage: float = 0.5,
    correlation_loss_weight: float | None = None,
    unsupported_groups: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Aggregate raw counts and write a target manifest with direct gene supervision."""
    if not 0 < minimum_gene_coverage <= 1:
        raise ValueError("minimum_gene_coverage must be in (0, 1].")
    if correlation_loss_weight is not None and correlation_loss_weight < 0:
        raise ValueError("correlation_loss_weight must be non-negative.")
    config = json.loads(targets_path.read_text())
    rna_head = _rna_head(config)
    target_groups = tuple(str(target["label"]) for target in rna_head["targets"])
    if len(set(target_groups)) != len(target_groups):
        raise ValueError("RNA target labels must be unique.")
    unsupported = set(unsupported_groups)
    unknown_unsupported = unsupported - set(target_groups)
    if unknown_unsupported:
        raise ValueError(
            f"Unsupported groups are not published targets: {sorted(unknown_unsupported)}."
        )
    supported_groups = tuple(group for group in target_groups if group not in unsupported)
    if not supported_groups:
        raise ValueError("At least one published group must retain direct-gene supervision.")

    barcode_groups, excluded_groups = read_barcode_groups(
        metadata_path,
        target_groups=supported_groups,
    )
    supported_counts = aggregate_matrix_market_by_group(
        matrix_path,
        barcode_path,
        barcode_groups,
        supported_groups,
    )
    feature_ids = read_10x_features(feature_path)
    if supported_counts.shape != (len(supported_groups), len(feature_ids)):
        raise ValueError(
            f"Aggregated count shape {supported_counts.shape} does not match "
            f"{len(supported_groups)} supported groups and {len(feature_ids)} features."
        )
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("Expression feature identifiers must be unique.")
    supported_cpm = normalize_counts_per_million(supported_counts)
    cpm = np.zeros((len(target_groups), len(feature_ids)), dtype=np.float32)
    supported_index = {group: index for index, group in enumerate(supported_groups)}
    group_valid = np.asarray([group in supported_index for group in target_groups])
    for output_index, group in enumerate(target_groups):
        if group_valid[output_index]:
            cpm[output_index] = supported_cpm[supported_index[group]]
    expression = PseudobulkExpression(
        groups=target_groups,
        gene_ids=feature_ids,
        cpm=cpm,
    )
    chromosome_sizes = build_fasta_index(fasta_path)
    genes = read_gene_exons(
        gtf_path,
        gene_ids=expression.gene_ids,
        chromosome_sizes=chromosome_sizes,
        gene_attribute="gene_name",
        strip_gene_versions=False,
    )
    coverage = len(genes) / len(expression.gene_ids)
    if coverage < minimum_gene_coverage:
        raise ValueError(
            f"Only {len(genes)}/{len(expression.gene_ids)} expression features "
            f"({coverage:.1%}) matched exon annotations and FASTA chromosomes."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    supervision_path = output_dir / "gene_expression_supervision.npz"
    matched_genes = write_gene_expression_supervision(
        supervision_path,
        expression,
        genes=genes,
        strip_gene_versions=False,
        group_valid=group_valid,
    )
    output_config = copy.deepcopy(config)
    output_rna_head = _rna_head(output_config)
    output_rna_head["gene_supervision"] = {
        "path": str(supervision_path.resolve()),
        "loss_weight": 1.0,
        "coverage_loss_weight": 1.0,
    }
    if correlation_loss_weight is not None:
        output_rna_head["double_centered_correlation_loss_weight"] = (
            correlation_loss_weight
        )
    contract = output_config.setdefault("target_contract", {})
    contract["rna_gene"] = "raw UMI counts summed by subclass, then counts per million"
    output_targets = output_dir / "targets.json"
    output_targets.write_text(json.dumps(output_config, indent=2) + "\n")

    group_counts = {group: 0 for group in target_groups}
    for group in barcode_groups.values():
        group_counts[group] += 1
    manifest = {
        "species": species,
        "matrix": str(matrix_path.resolve()),
        "barcodes": str(barcode_path.resolve()),
        "features": str(feature_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "source_targets": str(targets_path.resolve()),
        "gtf": str(gtf_path.resolve()),
        "fasta": str(fasta_path.resolve()),
        "normalization": "raw UMI counts summed by subclass, then counts per million",
        "published_groups": list(target_groups),
        "direct_gene_groups": list(supported_groups),
        "unsupported_direct_gene_groups": sorted(unsupported),
        "cells_per_group": group_counts,
        "excluded_groups": excluded_groups,
        "input_genes": len(feature_ids),
        "matched_genes": matched_genes,
        "matched_gene_fraction": coverage,
        "coverage_loss_weight": 1.0,
        "gene_loss_weight": 1.0,
        "double_centered_correlation_loss_weight": correlation_loss_weight,
        "targets": str(output_targets.resolve()),
        "gene_supervision": str(supervision_path.resolve()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--barcodes", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--gtf", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--species", required=True)
    parser.add_argument("--minimum-gene-coverage", type=float, default=0.5)
    parser.add_argument("--correlation-loss-weight", type=float)
    parser.add_argument("--unsupported-group", action="append", default=[])
    args = parser.parse_args()
    manifest = prepare_gene_supervision(
        matrix_path=args.matrix.expanduser().resolve(),
        barcode_path=args.barcodes.expanduser().resolve(),
        feature_path=args.features.expanduser().resolve(),
        metadata_path=args.metadata.expanduser().resolve(),
        targets_path=args.targets.expanduser().resolve(),
        gtf_path=args.gtf.expanduser().resolve(),
        fasta_path=args.fasta.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        species=args.species,
        minimum_gene_coverage=args.minimum_gene_coverage,
        correlation_loss_weight=args.correlation_loss_weight,
        unsupported_groups=tuple(args.unsupported_group),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
