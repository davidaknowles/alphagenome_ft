"""Aggregate sparse single-cell expression matrices into pseudobulks."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.io import mmread


def normalize_counts_per_million(counts: np.ndarray) -> np.ndarray:
    """Normalize a non-negative groups-by-genes count matrix to CPM."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Counts must be a finite non-negative groups-by-genes matrix.")
    totals = counts.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Every group must have positive total counts.")
    return (counts / totals * 1_000_000.0).astype(np.float32)


def aggregate_sparse_counts_by_group(
    counts: sparse.spmatrix,
    cell_groups: Sequence[str],
    groups: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Sum a cells-by-genes sparse count matrix by cell group.

    The returned tuple contains group labels, a dense groups-by-genes count
    matrix, and the number of cells assigned to each group.
    """
    if not sparse.issparse(counts):
        raise TypeError("counts must be a SciPy sparse matrix.")
    counts = counts.tocsr()
    if counts.shape[0] != len(cell_groups):
        raise ValueError(
            f"Count matrix has {counts.shape[0]} cells but {len(cell_groups)} labels were given."
        )
    if np.any(~np.isfinite(counts.data)) or np.any(counts.data < 0):
        raise ValueError("Sparse counts must be finite and non-negative.")

    cell_groups = tuple(str(group) for group in cell_groups)
    if groups is None:
        groups = tuple(sorted(set(cell_groups)))
    else:
        groups = tuple(str(group) for group in groups)
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("Groups must be nonempty and unique.")
    group_indices = {group: index for index, group in enumerate(groups)}
    unknown = sorted(set(cell_groups) - set(group_indices))
    if unknown:
        raise ValueError(f"Cell labels contain unknown groups: {unknown[:10]}")

    columns = np.fromiter(
        (group_indices[group] for group in cell_groups),
        dtype=np.int64,
        count=len(cell_groups),
    )
    rows = np.arange(len(cell_groups), dtype=np.int64)
    assignment = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(rows), len(groups)),
    )
    aggregated = assignment.T @ counts
    n_cells = np.bincount(columns, minlength=len(groups)).astype(np.int64)
    return groups, np.asarray(aggregated.toarray(), dtype=np.float64), n_cells


def align_cpm_to_gene_supervision(
    *,
    template_groups: Sequence[str],
    template_gene_ids: Sequence[str],
    source_groups: Sequence[str],
    source_gene_ids: Sequence[str],
    source_cpm: np.ndarray,
    source_gene_by_template: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Align pseudobulk expression to an existing supervision artifact.

    Values are renormalized after selecting the modeled gene set, matching the
    historical joint-target convention that CPM sums to one million over the
    genes represented by the supervision artifact.
    """
    template_groups = tuple(str(value) for value in template_groups)
    template_gene_ids = tuple(str(value) for value in template_gene_ids)
    source_groups = tuple(str(value) for value in source_groups)
    source_gene_ids = tuple(str(value) for value in source_gene_ids)
    source_cpm = np.asarray(source_cpm, dtype=np.float64)
    if source_cpm.shape != (len(source_groups), len(source_gene_ids)):
        raise ValueError("Source CPM shape does not match its group and gene labels.")
    if len(set(source_groups)) != len(source_groups):
        raise ValueError("Source group labels must be unique.")
    if len(set(source_gene_ids)) != len(source_gene_ids):
        raise ValueError("Source gene identifiers must be unique.")

    group_index = {group: index for index, group in enumerate(source_groups)}
    gene_index = {gene_id: index for index, gene_id in enumerate(source_gene_ids)}
    missing_groups = sorted(set(template_groups) - set(group_index))
    if missing_groups:
        raise ValueError(f"Source expression lacks groups: {missing_groups[:10]}")
    source_gene_by_template = source_gene_by_template or {}
    mapped_genes = tuple(
        source_gene_by_template.get(gene_id, gene_id) for gene_id in template_gene_ids
    )
    missing_genes = sorted(set(mapped_genes) - set(gene_index))
    if missing_genes:
        raise ValueError(f"Source expression lacks mapped genes: {missing_genes[:10]}")

    rows = np.asarray([group_index[group] for group in template_groups], dtype=np.int64)
    columns = np.asarray([gene_index[gene] for gene in mapped_genes], dtype=np.int64)
    return normalize_counts_per_million(source_cpm[np.ix_(rows, columns)])


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
    "aggregate_sparse_counts_by_group",
    "aggregate_matrix_market_by_group",
    "align_cpm_to_gene_supervision",
    "normalize_counts_per_million",
    "read_10x_barcodes",
    "read_10x_features",
]
