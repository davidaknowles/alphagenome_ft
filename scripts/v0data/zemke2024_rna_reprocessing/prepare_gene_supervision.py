#!/usr/bin/env python3
"""Build raw-count gene supervision for released Zemke 2024 broad classes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

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
    aggregate_10x_h5_columns_by_group,
    normalize_counts_per_million,
)

UNRELEASED_SUBTYPE_TARGETS = frozenset(
    {"Astro1_all", "Astro2_all", "Micro1_all", "Micro2_all"}
)


def _rna_head(config: dict[str, Any]) -> dict[str, Any]:
    heads = [head for head in config.get("heads", ()) if head.get("kind") == "rna_seq"]
    if len(heads) != 1:
        raise ValueError(f"Expected one RNA-seq head, found {len(heads)}.")
    return heads[0]


def target_groups_and_validity(config: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray]:
    """Return published target order and direct-gene validity."""
    groups = tuple(str(target["label"]) for target in _rna_head(config)["targets"])
    if len(groups) != len(set(groups)):
        raise ValueError("RNA target labels must be unique.")
    valid = np.asarray([group not in UNRELEASED_SUBTYPE_TARGETS for group in groups])
    return groups, valid


def _metadata_by_donor(
    metadata_path: Path, valid_groups: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    metadata = pd.read_csv(metadata_path, sep="\t")
    required = {"bacrode", "orig.ident", "subclass", "nCount_RNA"}
    if not required <= set(metadata):
        raise ValueError(f"Metadata lacks columns {sorted(required - set(metadata))}.")
    metadata = metadata.copy()
    metadata["target_group"] = metadata["subclass"].astype(str) + "_all"
    unknown = sorted(set(metadata["target_group"]) - set(valid_groups))
    if unknown:
        raise ValueError(f"Metadata contains groups absent from broad RNA targets: {unknown}.")
    return {
        str(donor): frame.copy()
        for donor, frame in metadata.groupby("orig.ident", sort=True)
    }


def prepare_gene_supervision(
    *,
    matrix_root: Path,
    metadata_path: Path,
    targets_path: Path,
    gtf_path: Path,
    fasta_path: Path,
    output_dir: Path,
    minimum_gene_coverage: float = 0.8,
    correlation_loss_weight: float | None = None,
) -> dict[str, Any]:
    """Aggregate filtered raw cells and attach partial direct-gene supervision."""
    config = json.loads(targets_path.read_text())
    target_groups, group_valid = target_groups_and_validity(config)
    valid_groups = tuple(group for group, valid in zip(target_groups, group_valid) if valid)
    metadata_by_donor = _metadata_by_donor(metadata_path, valid_groups)
    matrix_paths = {
        path.parent.name: path
        for path in matrix_root.glob("*/*_raw_feature_bc_matrix.h5")
    }
    missing_matrices = sorted(set(metadata_by_donor) - set(matrix_paths))
    if missing_matrices:
        raise FileNotFoundError(f"Missing donor matrices: {missing_matrices}.")

    total_counts: np.ndarray | None = None
    total_cells = np.zeros(len(valid_groups), dtype=np.int64)
    metadata_molecules = np.zeros(len(valid_groups), dtype=np.float64)
    feature_ids: tuple[str, ...] | None = None
    feature_names: tuple[str, ...] | None = None
    group_index = {group: idx for idx, group in enumerate(valid_groups)}
    donor_rows = []
    for donor, metadata in metadata_by_donor.items():
        prefix = donor + "_"
        full_barcodes = metadata["bacrode"].astype(str)
        if not full_barcodes.str.startswith(prefix).all():
            raise ValueError(
                f"Donor {donor} metadata contains a barcode without prefix {prefix!r}."
            )
        bare_barcodes = full_barcodes.str[len(prefix) :]
        barcode_groups = dict(zip(bare_barcodes, metadata["target_group"], strict=True))
        if len(barcode_groups) != len(metadata):
            raise ValueError(f"Donor {donor} contains duplicate bare barcodes.")
        ids, names, counts, n_cells = aggregate_10x_h5_columns_by_group(
            matrix_paths[donor], barcode_groups, valid_groups
        )
        if feature_ids is None:
            feature_ids, feature_names = ids, names
            total_counts = np.zeros_like(counts)
        elif ids != feature_ids or names != feature_names:
            raise ValueError(f"Donor {donor} has a different gene feature order.")
        total_counts += counts
        total_cells += n_cells
        donor_metadata_molecules = np.zeros(len(valid_groups), dtype=np.float64)
        for group, value in metadata.groupby("target_group")["nCount_RNA"].sum().items():
            donor_metadata_molecules[group_index[str(group)]] = float(value)
        if not np.allclose(counts.sum(axis=1), donor_metadata_molecules, rtol=0, atol=0):
            raise ValueError(f"Donor {donor} raw gene counts do not match metadata nCount_RNA.")
        metadata_molecules += donor_metadata_molecules
        donor_rows.append(
            {
                "donor": donor,
                "cells": int(n_cells.sum()),
                "rna_molecules": int(counts.sum()),
            }
        )
        print(
            f"aggregated {donor}, cells={n_cells.sum():,}, molecules={counts.sum():,.0f}",
            flush=True,
        )

    assert total_counts is not None and feature_ids is not None and feature_names is not None
    if not np.array_equal(total_counts.sum(axis=1), metadata_molecules):
        raise ValueError("Aggregated RNA molecule totals do not match release metadata.")
    broad_cpm = normalize_counts_per_million(total_counts)
    all_cpm = np.zeros((len(target_groups), len(feature_ids)), dtype=np.float32)
    all_cpm[group_valid] = broad_cpm
    expression = PseudobulkExpression(target_groups, feature_ids, all_cpm)

    chromosome_sizes = build_fasta_index(fasta_path)
    genes = read_gene_exons(
        gtf_path,
        gene_ids=feature_ids,
        chromosome_sizes=chromosome_sizes,
    )
    coverage = len(genes) / len(feature_ids)
    if coverage < minimum_gene_coverage:
        raise ValueError(
            f"Only {len(genes)}/{len(feature_ids)} RNA genes ({coverage:.1%}) matched annotations."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    supervision_path = output_dir / "gene_expression_supervision.npz"
    matched_genes = write_gene_expression_supervision(
        supervision_path,
        expression,
        genes=genes,
        group_valid=group_valid,
    )

    output_config = copy.deepcopy(config)
    output_rna = _rna_head(output_config)
    output_rna["gene_supervision"] = {
        "path": str(supervision_path.resolve()),
        "loss_weight": 1.0,
        "coverage_loss_weight": 1.0,
    }
    if correlation_loss_weight is not None:
        output_rna["double_centered_correlation_loss_weight"] = correlation_loss_weight
    output_config.setdefault("target_contract", {})["rna_gene"] = (
        "raw UMI counts from released filtered broad subclasses, summed then normalized "
        "to counts per million; unreleased Astro1/2 and Micro1/2 assignments are masked"
    )
    output_targets = output_dir / "targets.json"
    output_targets.write_text(json.dumps(output_config, indent=2) + "\n")
    manifest = {
        "dataset": "zemke2024-all",
        "source_targets": str(targets_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "matrix_root": str(matrix_root.resolve()),
        "donors": donor_rows,
        "published_groups": list(target_groups),
        "direct_gene_groups": list(valid_groups),
        "masked_gene_groups": [
            group for group, valid in zip(target_groups, group_valid) if not valid
        ],
        "cells_per_group": dict(zip(valid_groups, map(int, total_cells), strict=True)),
        "molecules_per_group": dict(
            zip(valid_groups, map(int, metadata_molecules), strict=True)
        ),
        "input_genes": len(feature_ids),
        "matched_genes": matched_genes,
        "matched_gene_fraction": coverage,
        "normalization": "raw UMI counts summed by broad subclass, then counts per million",
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
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--gtf", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-gene-coverage", type=float, default=0.8)
    parser.add_argument("--correlation-loss-weight", type=float)
    args = parser.parse_args()
    result = prepare_gene_supervision(
        matrix_root=args.matrix_root,
        metadata_path=args.metadata,
        targets_path=args.targets,
        gtf_path=args.gtf,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        minimum_gene_coverage=args.minimum_gene_coverage,
        correlation_loss_weight=args.correlation_loss_weight,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
