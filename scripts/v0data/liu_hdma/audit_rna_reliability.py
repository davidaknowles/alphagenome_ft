#!/usr/bin/env python
"""Estimate Liu RNA pseudobulk reliability from sample-level split halves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import (
    balanced_library_split,
    counts_per_million,
    double_centered_pearson,
    spearman_brown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    return parser.parse_args()


def _row_correlations(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(first * first, axis=1) * np.sum(second * second, axis=1))
    return np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.full((len(first),), np.nan),
        where=denominator > 0,
    )


def main() -> None:
    args = parse_args()
    paths = sorted(args.sample_dir.expanduser().resolve().glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No sample aggregates found under {args.sample_dir}.")

    groups = genes = shape = None
    library_sizes = []
    for path in paths:
        with np.load(path) as sample:
            sample_groups = tuple(sample["groups"].astype(str))
            sample_genes = tuple(sample["gene_ids"].astype(str))
            sample_shape = sample["counts"].shape
            if groups is None:
                groups, genes, shape = sample_groups, sample_genes, sample_shape
            elif (sample_groups, sample_genes, sample_shape) != (groups, genes, shape):
                raise ValueError(f"Sample axes differ in {path}.")
            library_sizes.append(sample["counts"].sum(axis=1, dtype=np.float64))
    assert groups is not None and genes is not None and shape is not None
    library_sizes = np.stack(library_sizes)
    first_assignment = balanced_library_split(library_sizes)

    first_counts = np.zeros(shape, dtype=np.float64)
    second_counts = np.zeros(shape, dtype=np.float64)
    for sample_index, path in enumerate(paths):
        with np.load(path) as sample:
            counts = np.asarray(sample["counts"], dtype=np.float64)
        first_mask = first_assignment[sample_index]
        first_counts[first_mask] += counts[first_mask]
        second_counts[~first_mask] += counts[~first_mask]

    first_cpm, first_valid = counts_per_million(first_counts)
    second_cpm, second_valid = counts_per_million(second_counts)
    valid = first_valid & second_valid
    first_cpm = first_cpm[valid]
    second_cpm = second_cpm[valid]
    raw_r = double_centered_pearson(first_cpm, second_cpm)
    log_r = double_centered_pearson(np.log1p(first_cpm), np.log1p(second_cpm))
    per_group_r = _row_correlations(np.log1p(first_cpm), np.log1p(second_cpm))
    quantiles = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    result = {
        "samples": len(paths),
        "groups": len(groups),
        "genes": len(genes),
        "groups_estimable_in_both_halves": int(valid.sum()),
        "raw_cpm_double_centered_r": raw_r,
        "raw_cpm_full_reliability_estimate": spearman_brown(raw_r),
        "log1p_cpm_double_centered_r": log_r,
        "log1p_cpm_full_reliability_estimate": spearman_brown(log_r),
        "log1p_per_group_pearson_quantiles": {
            str(quantile): float(np.nanquantile(per_group_r, quantile))
            for quantile in quantiles
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(
        "# Liu RNA split-half reliability\n\n"
        "Biological samples were assigned independently for each cell group to two greedily library-depth-balanced halves. Counts were summed and normalized to counts per million, CPM, within each half. The Spearman-Brown correction estimates reliability of the complete pseudobulk from equal-half correlation.\n\n"
        "| Quantity | Value |\n|---|---:|\n"
        f"| Samples | {result['samples']} |\n"
        f"| Cell groups estimable in both halves | {result['groups_estimable_in_both_halves']} / {result['groups']} |\n"
        f"| Genes | {result['genes']} |\n"
        f"| Raw CPM split-half double-centered R | {raw_r:.4f} |\n"
        f"| Raw CPM estimated full-pseudobulk reliability | {result['raw_cpm_full_reliability_estimate']:.4f} |\n"
        f"| log1p CPM split-half double-centered R | {log_r:.4f} |\n"
        f"| log1p CPM estimated full-pseudobulk reliability | {result['log1p_cpm_full_reliability_estimate']:.4f} |\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
