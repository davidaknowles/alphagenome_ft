#!/usr/bin/env python
"""Measure raw Zemke 2024 ATAC pseudobulk split-half reliability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import double_centered_pearson, spearman_brown
from alphagenome_ft.finetune.reprocessing import (
    BinnedAtacAccumulator,
    depth_balanced_half_assignments,
    stream_tabix_fragments,
)
from scripts.zemke2024_atac_reprocessing.aggregate import (
    discover_fragment_paths,
    read_fragment_histogram,
    read_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--chromosome-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=100)
    parser.add_argument("--tabix", default="tabix")
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(quantile): float(np.quantile(values, quantile))
        for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
    }


def main() -> None:
    args = parse_args()
    paths = discover_fragment_paths(args.fragment_root, max_donors=None)
    groups_by_donor = read_metadata(args.metadata)
    groups_by_path = {}
    histograms = {}
    missing_metadata = []
    for path in paths:
        groups = groups_by_donor.get(path.parent.name)
        if groups is None:
            missing_metadata.append(path.parent.name)
            continue
        groups_by_path[path] = groups
        histograms[path] = read_fragment_histogram(path)
    if missing_metadata:
        raise ValueError(f"No metadata for fragment donors: {', '.join(missing_metadata)}.")
    groups, assignments, half_depths, histogram_missing = depth_balanced_half_assignments(
        paths, histograms, groups_by_path
    )
    valid_groups = np.all(half_depths > 0, axis=1)
    if np.count_nonzero(valid_groups) < 2:
        raise ValueError("Split halves retain fewer than two nonempty cell groups.")
    accumulator = BinnedAtacAccumulator(
        num_groups=2 * len(groups),
        chromosome_size=args.chromosome_size,
        bin_size=args.bin_size,
    )
    matched_records = 0
    missing_records = 0
    for index, path in enumerate(paths, start=1):
        matched, missing = stream_tabix_fragments(
            path,
            args.chromosome,
            cell_group_indices=assignments[path],
            accumulator=accumulator,
            tabix=args.tabix,
        )
        matched_records += matched
        missing_records += missing
        print(f"[{index}/{len(paths)}] {path.parent.name}, matched={matched:,}, missing={missing:,}", flush=True)
    _, coverage = accumulator.normalized(np.maximum(half_depths.reshape(-1), 1.0))
    first = coverage[0::2][valid_groups].T
    second = coverage[1::2][valid_groups].T
    split_half_r = double_centered_pearson(first, second)
    full_reliability = spearman_brown(split_half_r)
    model_ceiling = float(np.sqrt(full_reliability)) if full_reliability >= 0 else float("nan")
    balance = np.min(half_depths, axis=1) / np.max(half_depths, axis=1)
    result = {
        "dataset": "zemke2024",
        "chromosome": args.chromosome,
        "chromosome_size": args.chromosome_size,
        "bin_size": args.bin_size,
        "signal": "coverage",
        "fragment_files": len(paths),
        "groups": len(groups),
        "groups_estimable_in_both_halves": int(np.count_nonzero(valid_groups)),
        "half_depth_quantiles": quantiles(half_depths[valid_groups].reshape(-1)),
        "within_group_half_depth_ratio_quantiles": quantiles(balance[valid_groups]),
        "split_half_double_centered_r": split_half_r,
        "full_target_reliability_estimate": full_reliability,
        "model_correlation_ceiling_estimate": model_ceiling,
        "histogram_cells_missing_metadata": histogram_missing,
        "chromosome_records_matched": matched_records,
        "chromosome_records_missing_metadata": missing_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
