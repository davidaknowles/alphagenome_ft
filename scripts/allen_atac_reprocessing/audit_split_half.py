#!/usr/bin/env python3
"""Measure fragment-derived ATAC pseudobulk split-half reliability on one chromosome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import (
    balanced_group_item_split,
    double_centered_pearson,
    spearman_brown,
)
from alphagenome_ft.finetune.reprocessing import (
    BinnedAtacAccumulator,
    match_fragment_library,
    read_cell_groups,
    read_cell_groups_by_library,
    read_fragment_histogram,
    stream_tabix_fragments,
)
from scripts.allen_atac_reprocessing.aggregate import discover_fragment_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-root", type=Path, required=True)
    parser.add_argument("--cell-metadata-h5ad", type=Path, required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--chromosome-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--fragment-library-suffix")
    parser.add_argument("--tabix", default="tabix")
    parser.add_argument("--fragment-cell-id-mode", choices=("direct", "library_barcode"), default="direct")
    parser.add_argument("--signal", choices=("coverage", "insertion"), default="coverage")
    return parser.parse_args()


def load_cell_groups(
    fragment_paths: list[Path],
    metadata_path: Path,
    *,
    cell_id_mode: str,
) -> tuple[dict[Path, dict[str, int]], dict[Path, dict[str, str]], dict[str, int]]:
    histograms = {
        path: read_fragment_histogram(path.parent / "statistics" / "histogram_cell.json")
        for path in fragment_paths
    }
    if cell_id_mode == "direct":
        groups = read_cell_groups(metadata_path)
        return histograms, {path: groups for path in fragment_paths}, {}
    groups_by_library = read_cell_groups_by_library(metadata_path)
    groups_by_path = {}
    matched_libraries = {}
    for path, histogram in histograms.items():
        library, groups = match_fragment_library(histogram, groups_by_library)
        groups_by_path[path] = groups
        matched_libraries[path.parent.name] = library
    return histograms, groups_by_path, matched_libraries


def build_half_assignments(
    fragment_paths: list[Path],
    histograms: dict[Path, dict[str, int]],
    groups_by_path: dict[Path, dict[str, str]],
) -> tuple[list[str], dict[Path, dict[str, int]], np.ndarray, int]:
    group_names = sorted({group for groups in groups_by_path.values() for group in groups.values()})
    group_index = {group: index for index, group in enumerate(group_names)}
    items_by_group: dict[str, list[tuple[Path, str, int]]] = {group: [] for group in group_names}
    histogram_cells_missing_metadata = 0
    for path in fragment_paths:
        cell_groups = groups_by_path[path]
        for cell, count in histograms[path].items():
            group = cell_groups.get(cell)
            if group is None:
                histogram_cells_missing_metadata += 1
                continue
            items_by_group[group].append((path, cell, count))

    assignments: dict[Path, dict[str, int]] = {path: {} for path in fragment_paths}
    half_depths = np.zeros((len(group_names), 2), dtype=np.float64)
    for group, items in items_by_group.items():
        if not items:
            continue
        item_groups = np.repeat(group, len(items))
        weights = np.asarray([item[2] for item in items], dtype=np.float64)
        keys = np.asarray([f"{item[0]}\0{item[1]}" for item in items])
        first_half = balanced_group_item_split(item_groups, weights, keys)
        group_offset = 2 * group_index[group]
        for item, first in zip(items, first_half, strict=True):
            path, cell, count = item
            half = 0 if first else 1
            assignments[path][cell] = group_offset + half
            half_depths[group_index[group], half] += count
    return group_names, assignments, half_depths, histogram_cells_missing_metadata


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(quantile): float(np.quantile(values, quantile))
        for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
    }


def main() -> None:
    args = parse_args()
    if args.chromosome_size < 1 or args.bin_size < 1 or args.chunk_size < 1:
        raise ValueError("Chromosome size, bin size, and chunk size must be positive.")
    fragment_paths = discover_fragment_paths(
        args.fragment_root,
        args.max_files,
        args.fragment_library_suffix,
    )
    histograms, groups_by_path, fragment_libraries = load_cell_groups(
        fragment_paths,
        args.cell_metadata_h5ad,
        cell_id_mode=args.fragment_cell_id_mode,
    )
    groups, assignments, half_depths, histogram_cells_missing_metadata = build_half_assignments(
        fragment_paths,
        histograms,
        groups_by_path,
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
    for index, path in enumerate(fragment_paths, start=1):
        matched, missing = stream_tabix_fragments(
            path,
            args.chromosome,
            cell_group_indices=assignments[path],
            accumulator=accumulator,
            chunk_size=args.chunk_size,
            tabix=args.tabix,
        )
        matched_records += matched
        missing_records += missing
        print(f"[{index}/{len(fragment_paths)}] {path.parent.name}, matched={matched:,}, missing={missing:,}", flush=True)
    normalization_depths = np.maximum(half_depths.reshape(-1), 1.0)
    insertion, coverage = accumulator.normalized(normalization_depths)
    signal = coverage if args.signal == "coverage" else insertion
    first = signal[0::2][valid_groups].T
    second = signal[1::2][valid_groups].T
    split_half_r = double_centered_pearson(first, second)
    full_reliability = spearman_brown(split_half_r)
    model_ceiling = float(np.sqrt(full_reliability)) if full_reliability >= 0 else float("nan")
    balance = np.min(half_depths, axis=1) / np.max(half_depths, axis=1)
    result = {
        "species": args.fragment_root.name,
        "chromosome": args.chromosome,
        "chromosome_size": args.chromosome_size,
        "bin_size": args.bin_size,
        "signal": args.signal,
        "fragment_files": len(fragment_paths),
        "groups": len(groups),
        "groups_estimable_in_both_halves": int(np.count_nonzero(valid_groups)),
        "half_depth_quantiles": quantiles(half_depths[valid_groups].reshape(-1)),
        "within_group_half_depth_ratio_quantiles": quantiles(balance[valid_groups]),
        "split_half_double_centered_r": split_half_r,
        "full_target_reliability_estimate": full_reliability,
        "model_correlation_ceiling_estimate": model_ceiling,
        "histogram_cells_missing_metadata": histogram_cells_missing_metadata,
        "chromosome_records_matched": matched_records,
        "chromosome_records_missing_metadata": missing_records,
        "fragment_cell_id_mode": args.fragment_cell_id_mode,
        "fragment_libraries": fragment_libraries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
