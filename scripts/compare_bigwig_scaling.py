#!/usr/bin/env python
"""Compare signal scaling across collections of BigWig tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyBigWig


def _quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0)
    quantiles = np.quantile(values, levels)
    return {f"q{level:g}": float(value) for level, value in zip(levels, quantiles)}


def summarize_collection(
    paths: list[Path],
    *,
    chromosome: str,
    starts: list[int],
    window_size: int,
    bin_size: int,
) -> dict[str, object]:
    if not paths:
        raise ValueError("BigWig collection is empty.")
    if window_size % bin_size:
        raise ValueError("Window size must be divisible by bin size.")

    binned_values: list[np.ndarray] = []
    track_means: list[float] = []
    track_rms: list[float] = []
    track_zero_fractions: list[float] = []
    total_count = 0
    total_zeros = 0
    total_sum = 0.0
    total_sum_squares = 0.0

    for path in paths:
        track_count = 0
        track_zeros = 0
        track_sum = 0.0
        track_sum_squares = 0.0
        with pyBigWig.open(str(path)) as handle:
            chrom_size = handle.chroms().get(chromosome)
            if chrom_size is None:
                raise ValueError(f"{path} does not contain {chromosome}.")
            for start in starts:
                end = start + window_size
                if start < 0 or end > chrom_size:
                    raise ValueError(
                        f"Window {chromosome}:{start}-{end} is outside {path} bounds."
                    )
                values = np.asarray(
                    handle.values(chromosome, start, end, numpy=True), dtype=np.float64
                )
                np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                track_count += values.size
                track_zeros += int(np.count_nonzero(values == 0.0))
                track_sum += float(values.sum())
                track_sum_squares += float(np.square(values).sum())
                binned_values.append(values.reshape(-1, bin_size).mean(axis=1))

        total_count += track_count
        total_zeros += track_zeros
        total_sum += track_sum
        total_sum_squares += track_sum_squares
        track_means.append(track_sum / track_count)
        track_rms.append(np.sqrt(track_sum_squares / track_count))
        track_zero_fractions.append(track_zeros / track_count)

    bins = np.concatenate(binned_values)
    nonzero_bins = bins[bins != 0.0]
    return {
        "tracks": len(paths),
        "sampled_bases": total_count,
        "base_zero_fraction": total_zeros / total_count,
        "base_mean": total_sum / total_count,
        "base_rms": float(np.sqrt(total_sum_squares / total_count)),
        "per_track_mean": _quantiles(np.asarray(track_means)),
        "per_track_rms": _quantiles(np.asarray(track_rms)),
        "per_track_zero_fraction": _quantiles(np.asarray(track_zero_fractions)),
        "binned_zero_fraction": float(np.mean(bins == 0.0)),
        "binned_values": _quantiles(bins),
        "binned_nonzero_values": _quantiles(nonzero_bins),
    }


def parse_collection(value: str) -> tuple[str, Path]:
    try:
        name, directory = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected NAME=DIRECTORY.") from exc
    if not name or not directory:
        raise argparse.ArgumentTypeError("Expected NAME=DIRECTORY.")
    return name, Path(directory).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        action="append",
        type=parse_collection,
        required=True,
        help="Named BigWig directory as NAME=DIRECTORY. May be repeated.",
    )
    parser.add_argument("--chromosome", default="chr8")
    parser.add_argument("--starts", default="5000000,21000000,37000000,53000000,69000000,85000000,101000000,117000000")
    parser.add_argument("--window-size", type=int, default=131072)
    parser.add_argument("--bin-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    starts = [int(value) for value in args.starts.split(",")]
    result = {
        "chromosome": args.chromosome,
        "starts": starts,
        "window_size": args.window_size,
        "bin_size": args.bin_size,
        "collections": {},
    }
    for name, directory in args.collection:
        paths = sorted(directory.glob("*.bw"))
        result["collections"][name] = summarize_collection(
            paths,
            chromosome=args.chromosome,
            starts=starts,
            window_size=args.window_size,
            bin_size=args.bin_size,
        )

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n")
    print(rendered)


if __name__ == "__main__":
    main()
