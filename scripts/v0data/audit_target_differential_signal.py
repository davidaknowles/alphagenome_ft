#!/usr/bin/env python3
"""Audit cross-track differential signal in genomic target manifests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyBigWig


def _canonical_chromosomes(
    chrom_sizes: dict[str, int], window_size: int, excluded: set[str]
) -> tuple[list[str], np.ndarray]:
    chromosomes = [
        chrom
        for chrom, size in chrom_sizes.items()
        if (chrom.startswith("NC_") or (chrom.startswith("chr") and "_" not in chrom))
        and chrom not in excluded
        and size >= window_size
    ]
    if not chromosomes:
        raise ValueError("No eligible canonical chromosomes found.")
    available_starts = np.asarray(
        [chrom_sizes[chrom] - window_size + 1 for chrom in chromosomes], dtype=np.float64
    )
    return chromosomes, available_starts / available_starts.sum()


def _sample_head(
    head: dict[str, Any],
    *,
    num_windows: int,
    window_size: int,
    num_bins: int,
    excluded_chromosomes: set[str],
    rng: np.random.Generator,
    workers: int,
) -> dict[str, Any]:
    targets = head["targets"]
    first_handle = pyBigWig.open(targets[0]["path"])
    try:
        chrom_sizes = first_handle.chroms()
    finally:
        first_handle.close()
    chromosomes, chromosome_probabilities = _canonical_chromosomes(
        chrom_sizes, window_size, excluded_chromosomes
    )
    regions = []
    for _ in range(num_windows):
        chromosome = str(rng.choice(chromosomes, p=chromosome_probabilities))
        start = int(rng.integers(0, chrom_sizes[chromosome] - window_size + 1))
        regions.append((chromosome, start))

    def read_target(target: dict[str, Any]) -> np.ndarray:
        handle = pyBigWig.open(target["path"])
        try:
            if handle.chroms() != chrom_sizes:
                raise ValueError(
                    f"Head {head['id']} contains tracks with different references."
                )
            windows = []
            for chromosome, start in regions:
                values = handle.stats(
                    chromosome,
                    start,
                    start + window_size,
                    nBins=num_bins,
                    type="mean",
                    exact=False,
                )
                windows.append(np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0))
            return np.stack(windows)
        finally:
            handle.close()

    if workers == 1:
        track_windows = [read_target(target) for target in targets]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(targets))
        ) as executor:
            track_windows = list(executor.map(read_target, targets))

    # values has dimensions [sampled genomic bins, target tracks].
    values = np.stack(track_windows, axis=2).reshape(-1, len(targets))
    centered = values - values.mean()
    double_centered = (
        values
        - values.mean(axis=0, keepdims=True)
        - values.mean(axis=1, keepdims=True)
        + values.mean()
    )
    total_sum_squares = float(np.sum(np.square(centered)))
    correlation = np.corrcoef(values, rowvar=False)
    upper_triangle = correlation[np.triu_indices(correlation.shape[0], k=1)]
    return {
        "id": head["id"],
        "kind": head["kind"],
        "num_tracks": values.shape[1],
        "num_observations": values.shape[0],
        "nonzero_fraction": float(np.mean(values > 0)),
        "standard_deviation": float(np.std(values)),
        "double_centered_standard_deviation": float(np.std(double_centered)),
        "double_centered_variance_fraction": (
            float(np.sum(np.square(double_centered)) / total_sum_squares)
            if total_sum_squares > 0
            else 0.0
        ),
        "median_pairwise_track_correlation": float(np.nanmedian(upper_triangle)),
    }


def audit_manifest(
    path: Path,
    *,
    num_windows: int,
    window_size: int,
    num_bins: int,
    excluded_chromosomes: set[str],
    seed: int,
    workers: int = 1,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    rng = np.random.default_rng(seed)
    return {
        "manifest": str(path),
        "dataset": manifest.get("dataset", path.parent.name),
        "num_windows": num_windows,
        "window_size": window_size,
        "num_bins_per_window": num_bins,
        "heads": [
            _sample_head(
                head,
                num_windows=num_windows,
                window_size=window_size,
                num_bins=num_bins,
                excluded_chromosomes=excluded_chromosomes,
                rng=rng,
                workers=workers,
            )
            for head in manifest["heads"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--num-windows", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=131_072)
    parser.add_argument("--num-bins", type=int, default=1024)
    parser.add_argument("--exclude-chromosomes", default="chr8,chr9")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.num_windows <= 0
        or args.window_size <= 0
        or args.num_bins <= 0
        or args.workers <= 0
    ):
        parser.error("Window, bin, and worker counts must be positive.")
    excluded = {item for item in args.exclude_chromosomes.split(",") if item}
    results = {
        "audits": [
            audit_manifest(
                path,
                num_windows=args.num_windows,
                window_size=args.window_size,
                num_bins=args.num_bins,
                excluded_chromosomes=excluded,
                seed=args.seed,
                workers=args.workers,
            )
            for path in args.manifests
        ]
    }
    rendered = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
