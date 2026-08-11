"""Aggregate sparse single-cell expression matrices into pseudobulks."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.io import mmread


def read_10x_barcodes(path: Path) -> tuple[str, ...]:
    """Read one barcode per line from a plain or gzip-compressed 10x file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return tuple(line.rstrip("\n") for line in handle)


def read_10x_features(path: Path) -> tuple[str, ...]:
    """Read stable feature identifiers from the first column of a 10x file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return tuple(line.rstrip("\n").split("\t", 1)[0] for line in handle)


def aggregate_matrix_market_by_group(
    matrix_path: Path,
    barcode_path: Path,
    barcode_groups: Mapping[str, str],
    groups: Sequence[str],
) -> np.ndarray:
    """Sum a genes-by-cells Matrix Market matrix into a groups-by-genes array.

    The assignment matrix has shape ``(C, K)``, where ``C`` is the number of
    matrix columns and ``K`` is the number of requested groups. It contains one
    nonzero per retained cell, allowing SciPy's sparse matrix multiplication to
    perform the aggregation without materializing a cell-by-gene dense array.
    """
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("Groups must be nonempty and unique.")
    group_indices = {group: index for index, group in enumerate(groups)}
    unknown = sorted(set(barcode_groups.values()) - set(group_indices))
    if unknown:
        raise ValueError(f"Barcode assignments contain unknown groups: {unknown[:10]}")

    barcodes = read_10x_barcodes(barcode_path)
    rows: list[int] = []
    columns: list[int] = []
    for cell_index, barcode in enumerate(barcodes):
        group = barcode_groups.get(barcode)
        if group is not None:
            rows.append(cell_index)
            columns.append(group_indices[group])
    if len(rows) != len(barcode_groups):
        barcode_set = set(barcodes)
        missing = sorted(set(barcode_groups) - barcode_set)
        raise ValueError(
            f"Only {len(rows)}/{len(barcode_groups)} assigned barcodes occur in the matrix; "
            f"examples missing: {missing[:10]}"
        )

    matrix = mmread(str(matrix_path))
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    if matrix.shape[1] != len(barcodes):
        raise ValueError(
            f"Matrix has {matrix.shape[1]} cells but {barcode_path} has {len(barcodes)} barcodes."
        )
    assignment = sparse.csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=(len(barcodes), len(groups)),
    )
    aggregated = matrix.tocsr() @ assignment
    return np.asarray(aggregated.toarray().T, dtype=np.float64)


__all__ = [
    "aggregate_matrix_market_by_group",
    "read_10x_barcodes",
    "read_10x_features",
]
