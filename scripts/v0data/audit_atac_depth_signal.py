#!/usr/bin/env python
"""Relate ATAC pseudobulk fragment depth to normalized target structure."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyBigWig

from scripts.v0data.audit_target_differential_signal import _canonical_chromosomes


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def summarize_relationship(
    labels: list[str],
    values: np.ndarray,
    depths: np.ndarray,
) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("Target values must have dimensions [observations, tracks].")
    if depths.shape != (len(labels),) or np.any(depths <= 0):
        raise ValueError("Every target track must have one positive fragment depth.")
    rms = np.sqrt(np.mean(np.square(values), axis=0))
    nonzero = np.mean(values > 0, axis=0)
    correlations = np.corrcoef(values, rowvar=False)
    median_other_correlation = np.asarray(
        [np.nanmedian(np.delete(correlations[index], index)) for index in range(len(labels))]
    )
    log_depth = np.log10(depths)
    log_rms = np.log10(np.maximum(rms, np.finfo(np.float64).tiny))
    return {
        "num_tracks": len(labels),
        "depth_quantiles": {
            str(quantile): float(np.quantile(depths, quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "groups_below_5m": int(np.sum(depths < 5_000_000)),
        "groups_below_10m": int(np.sum(depths < 10_000_000)),
        "groups_below_25m": int(np.sum(depths < 25_000_000)),
        "spearman_log_depth_vs_log_rms": _spearman(log_depth, log_rms),
        "spearman_log_depth_vs_nonzero_fraction": _spearman(log_depth, nonzero),
        "spearman_log_depth_vs_median_track_correlation": _spearman(
            log_depth, median_other_correlation
        ),
        "tracks": [
            {
                "label": label,
                "fragments": int(depth),
                "rms": float(track_rms),
                "nonzero_fraction": float(track_nonzero),
                "median_other_track_correlation": float(track_correlation),
            }
            for label, depth, track_rms, track_nonzero, track_correlation in zip(
                labels,
                depths,
                rms,
                nonzero,
                median_other_correlation,
                strict=True,
            )
        ],
    }


def audit_manifest(
    manifest_path: Path,
    depth_path: Path,
    *,
    head_id: str,
    num_windows: int,
    window_size: int,
    num_bins: int,
    excluded_chromosomes: set[str],
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if window_size % num_bins:
        raise ValueError("Window size must be divisible by the number of bins.")
    manifest = json.loads(manifest_path.read_text())
    heads = [head for head in manifest["heads"] if head["id"] == head_id]
    if len(heads) != 1:
        raise ValueError(f"Expected one head named {head_id!r} in {manifest_path}.")
    targets = heads[0]["targets"]
    labels = [str(target["label"]) for target in targets]
    with np.load(depth_path, allow_pickle=False) as depth_file:
        depth_by_group = dict(
            zip(
                depth_file["groups"].astype(str).tolist(),
                depth_file["total_fragments"].astype(np.float64).tolist(),
                strict=True,
            )
        )
    missing = [label for label in labels if label not in depth_by_group]
    if missing:
        raise ValueError(f"Depth file lacks target groups {missing}.")
    depths = np.asarray([depth_by_group[label] for label in labels], dtype=np.float64)

    with pyBigWig.open(targets[0]["path"]) as handle:
        chrom_sizes = handle.chroms()
    chromosomes, probabilities = _canonical_chromosomes(
        chrom_sizes,
        window_size,
        excluded_chromosomes,
    )
    rng = np.random.default_rng(seed)
    regions = []
    for _ in range(num_windows):
        chromosome = str(rng.choice(chromosomes, p=probabilities))
        start = int(rng.integers(0, chrom_sizes[chromosome] - window_size + 1))
        regions.append((chromosome, start))
    bin_width = window_size // num_bins

    def read_target(target: dict[str, Any]) -> np.ndarray:
        windows = []
        with pyBigWig.open(target["path"]) as handle:
            for chromosome, start in regions:
                values = handle.values(
                    chromosome,
                    start,
                    start + window_size,
                    numpy=True,
                )
                dense = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
                windows.append(dense.reshape(num_bins, bin_width).mean(axis=1))
        return np.concatenate(windows)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(targets))) as executor:
        values = np.stack(list(executor.map(read_target, targets)), axis=1)
    return {
        "manifest": str(manifest_path),
        "depths": str(depth_path),
        "head_id": head_id,
        "num_windows": num_windows,
        "window_size": window_size,
        "num_bins_per_window": num_bins,
        **summarize_relationship(labels, values, depths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--depths", type=Path, required=True)
    parser.add_argument("--head-id", required=True)
    parser.add_argument("--num-windows", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=131_072)
    parser.add_argument("--num-bins", type=int, default=1024)
    parser.add_argument("--exclude-chromosomes", default="chr8,chr9")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_manifest(
        args.manifest,
        args.depths,
        head_id=args.head_id,
        num_windows=args.num_windows,
        window_size=args.window_size,
        num_bins=args.num_bins,
        excluded_chromosomes={
            chromosome for chromosome in args.exclude_chromosomes.split(",") if chromosome
        },
        seed=args.seed,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
