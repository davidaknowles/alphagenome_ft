#!/usr/bin/env python
"""Aggregate raw single-cell UMI counts into group-level CPM pseudobulks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reprocessing import (
    aggregate_sparse_count_chunks_by_group,
    normalize_counts_per_million,
)


def _read_h5ad_column(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    if isinstance(node, h5py.Dataset):
        return node[:]
    if isinstance(node, h5py.Group) and {"categories", "codes"} <= set(node):
        categories = node["categories"][:]
        codes = node["codes"][:]
        if np.any(codes < 0):
            raise ValueError(f"{name} contains missing group labels.")
        return categories[codes]
    raise ValueError(f"Unsupported H5AD encoding for {group.name}/{name}.")


def _iter_h5ad_csr_rows(matrix: h5py.Group, chunk_size: int):
    encoding = matrix.attrs.get("encoding-type")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding != "csr_matrix":
        raise ValueError(f"{matrix.name} must use H5AD CSR encoding, not {encoding!r}.")
    n_rows, n_columns = (int(value) for value in matrix.attrs["shape"])
    indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
    if indptr.shape != (n_rows + 1,):
        raise ValueError(f"{matrix.name}/indptr does not match the encoded shape.")
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        data_start = int(indptr[start])
        data_stop = int(indptr[stop])
        local_indptr = indptr[start : stop + 1] - data_start
        yield sparse.csr_matrix(
            (
                matrix["data"][data_start:data_stop],
                matrix["indices"][data_start:data_stop],
                local_indptr,
            ),
            shape=(stop - start, n_columns),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-column", default="Group")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    with h5py.File(args.input.expanduser().resolve(), "r") as source:
        if "raw/X" not in source or "raw/var" not in source:
            raise ValueError("Input H5AD does not contain raw counts and annotations.")
        labels = tuple(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in _read_h5ad_column(source["obs"], args.group_column)
        )
        var = read_elem(source["raw/var"])
        groups, aggregated, n_cells = aggregate_sparse_count_chunks_by_group(
            _iter_h5ad_csr_rows(source["raw/X"], args.chunk_size), labels
        )

    totals = aggregated.sum(axis=1)
    cpm = normalize_counts_per_million(aggregated)
    obs = pd.DataFrame(
        {"n_cells": n_cells, "total_counts": totals},
        index=pd.Index(groups, name=args.group_column),
    )
    result = ad.AnnData(X=cpm, obs=obs, var=var)
    result.uns["normalization"] = "raw UMI counts summed by group, then counts per million"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(args.output)
    print(
        f"Wrote {len(groups)} groups x {result.n_vars} genes to {args.output}; "
        f"row sums {cpm.sum(axis=1).min():.1f}-{cpm.sum(axis=1).max():.1f}."
    )


if __name__ == "__main__":
    main()
