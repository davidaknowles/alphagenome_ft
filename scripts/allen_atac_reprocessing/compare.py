#!/usr/bin/env python
"""Compare reprocessed Allen ATAC shards with released group BigWigs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pyBigWig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--released-bigwig-dir", type=Path, required=True)
    parser.add_argument(
        "--distribution-reference-dir",
        type=Path,
        help="Optional BigWig collection summarized on the same windows.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", required=True, help="Comma-separated, bin-aligned starts")
    parser.add_argument("--window-size", type=int, default=131_000)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def released_bins(
    path: Path,
    chromosome: str,
    starts: list[int],
    window_size: int,
    bin_size: int,
) -> np.ndarray:
    chunks = []
    with pyBigWig.open(str(path)) as handle:
        for start in starts:
            values = np.asarray(
                handle.values(chromosome, start, start + window_size, numpy=True),
                dtype=np.float32,
            )
            np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            chunks.append(values.reshape(-1, bin_size).mean(axis=1))
    return np.concatenate(chunks)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x, dtype=np.float64)
    y_centered = y - np.mean(y, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(np.square(x_centered), dtype=np.float64)
        * np.sum(np.square(y_centered), dtype=np.float64)
    )
    return float(np.sum(x_centered * y_centered, dtype=np.float64) / denominator)


def double_center(values: np.ndarray) -> np.ndarray:
    return (
        values
        - values.mean(axis=0, keepdims=True)
        - values.mean(axis=1, keepdims=True)
        + values.mean()
    )


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values, dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "zero_fraction": float(np.mean(values == 0)),
        "q99": float(np.quantile(values, 0.99)),
    }


def main() -> None:
    args = parse_args()
    with np.load(args.shard) as shard:
        groups = shard["groups"].astype(str).tolist()
        chromosome = str(shard["chromosome"])
        bin_size = int(shard["bin_size"])
        strategies = {
            "insertion_spmr": shard["insertion_spmr"],
            "coverage_spmr": shard["coverage_spmr"],
        }
    starts = [int(value) for value in args.starts.split(",")]
    if args.window_size % bin_size or any(start % bin_size for start in starts):
        raise ValueError("Window size and starts must be aligned to the shard bin size.")
    bins_per_window = args.window_size // bin_size
    bin_indices = np.concatenate(
        [np.arange(start // bin_size, start // bin_size + bins_per_window) for start in starts]
    )

    released = []
    for group in groups:
        path = args.released_bigwig_dir / f"{safe_name(group)}.bw"
        if not path.exists():
            raise FileNotFoundError(f"Released BigWig not found for {group}, {path}")
        released.append(released_bins(path, chromosome, starts, args.window_size, bin_size))
    released_matrix = np.stack(released, axis=1)
    result = {"released": summarize(released_matrix), "strategies": {}}
    for name, values in strategies.items():
        matrix = values[:, bin_indices].T
        result["strategies"][name] = {
            **summarize(matrix),
            "released_pearson_r": pearson(matrix, released_matrix),
            "released_double_centered_r": pearson(
                double_center(matrix), double_center(released_matrix)
            ),
        }
    if args.distribution_reference_dir is not None:
        reference_paths = sorted(args.distribution_reference_dir.glob("*.bw"))
        if not reference_paths:
            raise FileNotFoundError(
                f"No reference BigWigs found in {args.distribution_reference_dir}."
            )
        reference_matrix = np.stack(
            [
                released_bins(path, chromosome, starts, args.window_size, bin_size)
                for path in reference_paths
            ],
            axis=1,
        )
        result["distribution_reference"] = {
            **summarize(reference_matrix),
            "tracks": len(reference_paths),
        }
    result["chromosome"] = chromosome
    result["starts"] = starts
    result["window_size"] = args.window_size
    result["bin_size"] = bin_size
    result["groups"] = len(groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
