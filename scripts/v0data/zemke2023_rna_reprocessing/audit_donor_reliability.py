#!/usr/bin/env python3
"""Estimate Zemke 2023 RNA reliability from donor-level split halves."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import (
    double_centered_pearson,
    spearman_brown,
    split_half_pseudobulks,
)
from alphagenome_ft.finetune.reprocessing import (
    aggregate_matrix_market_by_group,
    read_10x_features,
)

from scripts.v0data.zemke2023_rna_reprocessing.prepare_gene_supervision import (
    safe_group_name,
)


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def read_donor_group_labels(
    metadata_path: Path,
    *,
    valid_groups: tuple[str, ...],
    cell_column: str = "cell",
    donor_column: str = "orig.ident",
    group_column: str = "subclass",
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Map retained barcodes to donor-group labels."""
    normalized_groups = {safe_group_name(group): group for group in valid_groups}
    if len(normalized_groups) != len(valid_groups):
        raise ValueError("Valid group labels are not unique after normalization.")
    barcode_labels: dict[str, str] = {}
    donors: set[str] = set()
    with _open_text(metadata_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {cell_column, donor_column, group_column}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"{metadata_path} must contain {sorted(required)}.")
        for row in reader:
            group = normalized_groups.get(safe_group_name(row[group_column]))
            if group is None:
                continue
            barcode = row[cell_column]
            donor = row[donor_column]
            if not barcode or not donor:
                raise ValueError("Retained metadata rows require a cell and donor label.")
            if barcode in barcode_labels:
                raise ValueError(f"Duplicate metadata barcode {barcode!r}.")
            barcode_labels[barcode] = f"{donor}\x1f{group}"
            donors.add(donor)
    if not barcode_labels:
        raise ValueError("No metadata cells matched valid target groups.")
    return barcode_labels, tuple(sorted(donors))


def audit_reliability(
    *,
    matrix_path: Path,
    barcode_path: Path,
    feature_path: Path,
    metadata_path: Path,
    supervision_path: Path,
    species: str,
) -> dict[str, object]:
    """Aggregate donor pseudobulks and estimate complete-target reliability."""
    with np.load(supervision_path, allow_pickle=False) as supervision:
        groups = tuple(supervision["groups"].astype(str))
        group_valid = supervision["group_valid"].astype(bool)
        gene_ids = tuple(supervision["gene_ids"].astype(str))
    valid_groups = tuple(group for group, valid in zip(groups, group_valid, strict=True) if valid)
    barcode_labels, donors = read_donor_group_labels(
        metadata_path, valid_groups=valid_groups
    )
    combined_groups = tuple(
        f"{donor}\x1f{group}" for donor in donors for group in valid_groups
    )
    aggregated = aggregate_matrix_market_by_group(
        matrix_path,
        barcode_path,
        barcode_labels,
        combined_groups,
    )
    source_gene_ids = read_10x_features(feature_path)
    source_gene_index = {gene_id: index for index, gene_id in enumerate(source_gene_ids)}
    if len(source_gene_index) != len(source_gene_ids):
        raise ValueError("Expression feature identifiers must be unique.")
    missing = sorted(set(gene_ids) - set(source_gene_index))
    if missing:
        raise ValueError(f"Expression matrix lacks modeled genes: {missing[:10]}")
    columns = np.asarray([source_gene_index[gene_id] for gene_id in gene_ids])
    counts = aggregated[:, columns].reshape(
        len(donors), len(valid_groups), len(gene_ids)
    )

    first, second, estimable = split_half_pseudobulks(counts)
    first = first[estimable]
    second = second[estimable]
    raw_r = double_centered_pearson(first, second)
    log_r = double_centered_pearson(np.log1p(first), np.log1p(second))
    raw_reliability = spearman_brown(raw_r)
    donor_support = np.sum(counts.sum(axis=2) > 0, axis=0)
    return {
        "species": species,
        "donors": len(donors),
        "groups": len(valid_groups),
        "genes": len(gene_ids),
        "groups_estimable_in_both_halves": int(estimable.sum()),
        "donor_support_quantiles": {
            str(q): float(np.quantile(donor_support, q))
            for q in (0, 0.25, 0.5, 0.75, 1)
        },
        "raw_cpm_double_centered_r": raw_r,
        "raw_cpm_full_reliability_estimate": raw_reliability,
        "raw_cpm_model_correlation_ceiling_estimate": float(
            np.sqrt(max(raw_reliability, 0.0))
        ),
        "log1p_cpm_double_centered_r": log_r,
        "log1p_cpm_full_reliability_estimate": spearman_brown(log_r),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--barcodes", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--supervision", type=Path)
    parser.add_argument("--species", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        "matrix": args.matrix,
        "barcodes": args.barcodes,
        "features": args.features,
        "metadata": args.metadata,
        "supervision": args.supervision,
    }
    if args.manifest is not None:
        if any(path is not None for path in paths.values()):
            parser.error("--manifest cannot be combined with explicit input paths")
        manifest = json.loads(args.manifest.read_text())
        paths = {
            "matrix": Path(manifest["matrix"]),
            "barcodes": Path(manifest["barcodes"]),
            "features": Path(manifest["features"]),
            "metadata": Path(manifest["metadata"]),
            "supervision": Path(manifest["gene_supervision"]),
        }
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        parser.error(f"Missing input paths: {', '.join(missing)}")
    result = audit_reliability(
        matrix_path=paths["matrix"],
        barcode_path=paths["barcodes"],
        feature_path=paths["features"],
        metadata_path=paths["metadata"],
        supervision_path=paths["supervision"],
        species=args.species,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
