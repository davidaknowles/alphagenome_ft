#!/usr/bin/env python
"""Prepare Allen ATAC target transforms using training chromosomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyBigWig

QUANTILE_LEVELS = np.asarray(
    [*np.linspace(0.0, 0.95, 40), 0.97, 0.98, 0.99, 0.995, 0.999, 1.0],
    dtype=np.float64,
)


def read_samples(paths: list[Path], chromosome: str, starts: list[int], window_size: int):
    samples = []
    means = []
    rms = []
    for path in paths:
        chunks = []
        with pyBigWig.open(str(path)) as handle:
            for start in starts:
                values = np.asarray(
                    handle.values(chromosome, start, start + window_size, numpy=True),
                    dtype=np.float32,
                )
                np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                chunks.append(values)
        values = np.concatenate(chunks)
        samples.append(values)
        means.append(float(np.mean(values, dtype=np.float64)))
        rms.append(float(np.sqrt(np.mean(np.square(values), dtype=np.float64))))
    return samples, np.asarray(means), np.asarray(rms)


def monotone_quantile_knots(source: np.ndarray, reference: np.ndarray, blend: float):
    source_positive = source[source > 0]
    if source_positive.size == 0:
        raise ValueError("Cannot quantile-normalize a track with no positive samples.")
    source_quantiles = np.quantile(source_positive, QUANTILE_LEVELS)
    reference_quantiles = np.quantile(reference, QUANTILE_LEVELS)
    transformed_quantiles = (1.0 - blend) * source_quantiles + blend * reference_quantiles

    source_knots = [0.0]
    transformed_knots = [0.0]
    for source_value, transformed_value in zip(source_quantiles, transformed_quantiles):
        if source_value > source_knots[-1] and transformed_value > transformed_knots[-1]:
            source_knots.append(float(source_value))
            transformed_knots.append(float(transformed_value))
    if len(source_knots) < 2:
        raise ValueError("Quantile samples did not produce two strictly increasing knots.")
    return source_knots, transformed_knots


def targets(paths: list[Path], nonzero_means: np.ndarray | None = None):
    result = []
    for index, path in enumerate(paths):
        record = {"path": str(path.resolve()), "label": path.stem, "strand": "."}
        if nonzero_means is not None:
            record["nonzero_mean"] = float(nonzero_means[index])
        result.append(record)
    return result


def write_strategy(
    output_dir: Path,
    name: str,
    paths: list[Path],
    *,
    nonzero_means: np.ndarray | None = None,
    transform_path: Path | None = None,
    resolutions: list[int] | None = None,
):
    head = {
        "id": "allen_atac",
        "source": "predefined",
        "kind": "atac",
        "resolutions": resolutions or [1, 128],
        "apply_squashing": False,
        "targets": targets(paths, nonzero_means),
    }
    if transform_path is not None:
        head["target_transform"] = {"path": str(transform_path.resolve())}
    strategy_dir = output_dir / name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "targets.json").write_text(json.dumps({"heads": [head]}, indent=2) + "\n")


def write_spatial_strategy(
    output_dir: Path,
    name: str,
    paths: list[Path],
    *,
    width: int,
    output_scale: float,
):
    strategy_dir = output_dir / name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    transform_path = strategy_dir / "target_transform.json"
    transform_path.write_text(
        json.dumps(
            {
                "kind": "spatial_rebin",
                "width": width,
                "output_scale": output_scale,
            },
            indent=2,
        )
        + "\n"
    )
    write_strategy(output_dir, name, paths, transform_path=transform_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allen-dir", type=Path, required=True)
    parser.add_argument("--hda-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chromosome", default="chr1")
    parser.add_argument(
        "--starts",
        default="5000000,33000000,61000000,89000000,117000000,145000000,173000000,201000000",
    )
    parser.add_argument("--window-size", type=int, default=131072)
    args = parser.parse_args()

    allen_paths = sorted(args.allen_dir.expanduser().glob("*.bw"))
    hda_paths = sorted(args.hda_dir.expanduser().glob("*.bw"))
    if not allen_paths or not hda_paths:
        raise FileNotFoundError("Both input directories must contain BigWig tracks.")
    starts = [int(value) for value in args.starts.split(",")]
    allen_samples, allen_means, allen_rms = read_samples(
        allen_paths, args.chromosome, starts, args.window_size
    )
    hda_samples, hda_means, hda_rms = read_samples(
        hda_paths, args.chromosome, starts, args.window_size
    )
    hda_positive = np.concatenate([values[values > 0][::16] for values in hda_samples])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_allen_tracks = len(allen_paths)
    write_strategy(
        args.output_dir, "scale4", allen_paths, nonzero_means=np.full(num_allen_tracks, 0.25)
    )
    write_strategy(
        args.output_dir, "scale10", allen_paths, nonzero_means=np.full(num_allen_tracks, 0.1)
    )
    write_strategy(
        args.output_dir, "scale20", allen_paths, nonzero_means=np.full(num_allen_tracks, 0.05)
    )

    mean_factors = np.clip(np.median(hda_means) / allen_means, 2.0, 32.0)
    rms_factors = np.clip(np.median(hda_rms) / allen_rms, 2.0, 32.0)
    write_strategy(args.output_dir, "track_mean", allen_paths, nonzero_means=1.0 / mean_factors)
    write_strategy(args.output_dir, "track_rms", allen_paths, nonzero_means=1.0 / rms_factors)
    write_spatial_strategy(args.output_dir, "smooth100", allen_paths, width=100, output_scale=1.0)
    write_spatial_strategy(
        args.output_dir, "rebin100_sum", allen_paths, width=100, output_scale=10.0
    )
    write_strategy(args.output_dir, "resolution128", allen_paths, resolutions=[128])

    for name, blend in (("quantile50", 0.5), ("quantile100", 1.0)):
        source_knots = []
        transformed_knots = []
        for sample in allen_samples:
            source, transformed = monotone_quantile_knots(sample, hda_positive, blend)
            source_knots.append(source)
            transformed_knots.append(transformed)
        strategy_dir = args.output_dir / name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        transform_path = strategy_dir / "target_transform.json"
        transform_path.write_text(
            json.dumps(
                {
                    "kind": "piecewise_linear",
                    "source_knots": source_knots,
                    "transformed_knots": transformed_knots,
                },
                indent=2,
            )
            + "\n"
        )
        write_strategy(
            args.output_dir,
            name,
            allen_paths,
            transform_path=transform_path,
        )

    summary = {
        "chromosome": args.chromosome,
        "starts": starts,
        "window_size": args.window_size,
        "allen_tracks": len(allen_paths),
        "hda_tracks": len(hda_paths),
        "allen_mean_median": float(np.median(allen_means)),
        "allen_rms_median": float(np.median(allen_rms)),
        "hda_mean_median": float(np.median(hda_means)),
        "hda_rms_median": float(np.median(hda_rms)),
        "track_mean_factor_quantiles": np.quantile(mean_factors, [0, 0.5, 1]).tolist(),
        "track_rms_factor_quantiles": np.quantile(rms_factors, [0, 0.5, 1]).tolist(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
