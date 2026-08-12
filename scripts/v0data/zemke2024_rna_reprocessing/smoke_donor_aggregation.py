#!/usr/bin/env python3
"""Validate raw 10x aggregation for one Zemke 2024 donor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reprocessing import aggregate_10x_h5_columns_by_group
from scripts.v0data.zemke2024_rna_reprocessing.prepare_gene_supervision import (
    bare_barcodes_for_donor,
    target_groups_and_validity,
)


def smoke_donor(
    *,
    donor: str,
    matrix_path: Path,
    metadata_path: Path,
    targets_path: Path,
) -> dict[str, object]:
    """Aggregate one donor and require exact metadata count recovery."""
    config = json.loads(targets_path.read_text())
    target_groups, group_valid = target_groups_and_validity(config)
    valid_groups = tuple(group for group, valid in zip(target_groups, group_valid) if valid)
    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        usecols=["bacrode", "orig.ident", "subclass", "nCount_RNA"],
    )
    metadata = metadata.loc[metadata["orig.ident"].astype(str) == donor].copy()
    if metadata.empty:
        raise ValueError(f"Metadata contains no cells for donor {donor!r}.")
    metadata["target_group"] = metadata["subclass"].astype(str) + "_all"
    unknown = sorted(set(metadata["target_group"]) - set(valid_groups))
    if unknown:
        raise ValueError(f"Donor {donor} contains unsupported groups: {unknown}.")
    barcode_groups = dict(
        zip(
            bare_barcodes_for_donor(metadata["bacrode"], donor),
            metadata["target_group"],
            strict=True,
        )
    )
    if len(barcode_groups) != len(metadata):
        raise ValueError(f"Donor {donor} contains duplicate bare barcodes.")

    feature_ids, _, counts, n_cells = aggregate_10x_h5_columns_by_group(
        matrix_path, barcode_groups, valid_groups
    )
    expected_cells = metadata.groupby("target_group").size().reindex(valid_groups, fill_value=0)
    expected_molecules = (
        metadata.groupby("target_group")["nCount_RNA"]
        .sum()
        .reindex(valid_groups, fill_value=0)
    )
    if not np.array_equal(n_cells, expected_cells.to_numpy(dtype=np.int64)):
        raise ValueError(f"Donor {donor} retained cell counts do not match metadata.")
    observed_molecules = counts.sum(axis=1)
    if not np.array_equal(observed_molecules, expected_molecules.to_numpy(dtype=np.float64)):
        raise ValueError(f"Donor {donor} RNA molecule totals do not match metadata.")
    return {
        "donor": donor,
        "cells": int(n_cells.sum()),
        "rna_molecules": int(observed_molecules.sum()),
        "gene_features": len(feature_ids),
        "valid_groups": len(valid_groups),
        "nonempty_groups": int(np.count_nonzero(n_cells)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = smoke_donor(
        donor=args.donor,
        matrix_path=args.matrix,
        metadata_path=args.metadata,
        targets_path=args.targets,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
