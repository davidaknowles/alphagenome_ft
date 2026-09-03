#!/usr/bin/env python
"""Compare a raw Zemke 2024 ATAC shard with matching released broad tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyBigWig

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import double_centered_pearson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_released_bins(path: Path, chromosome: str, bin_size: int, num_bins: int) -> np.ndarray:
    with pyBigWig.open(str(path)) as handle:
        chromosome_size = handle.chroms().get(chromosome)
        if chromosome_size is None:
            raise ValueError(f"{path} does not contain {chromosome}.")
        values = np.asarray(handle.values(chromosome, 0, chromosome_size, numpy=True), dtype=np.float64)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    starts = np.arange(num_bins, dtype=np.int64) * bin_size
    ends = np.minimum(starts + bin_size, values.size)
    widths = ends - starts
    if np.any(widths <= 0):
        raise ValueError(f"{path} is shorter than the raw shard bins.")
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return ((cumulative[ends] - cumulative[starts]) / widths).astype(np.float32)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    centered_x = x - np.mean(x, dtype=np.float64)
    centered_y = y - np.mean(y, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(np.square(centered_x), dtype=np.float64)
        * np.sum(np.square(centered_y), dtype=np.float64)
    )
    if denominator == 0:
        return float("nan")
    return float(np.sum(centered_x * centered_y, dtype=np.float64) / denominator)


def released_paths(target_manifest: Path) -> dict[str, Path]:
    payload = json.loads(target_manifest.read_text())
    for head in payload["heads"]:
        if head["id"] == "zemke2024_all_atac":
            return {
                target["label"].removesuffix("_all"): Path(target["path"])
                for target in head["targets"]
            }
    raise ValueError(f"No zemke2024_all_atac head in {target_manifest}.")


def main() -> None:
    args = parse_args()
    with np.load(args.shard) as shard:
        groups = shard["groups"].astype(str).tolist()
        chromosome = str(shard["chromosome"])
        bin_size = int(shard["bin_size"])
        raw = np.asarray(shard["coverage_spmr"], dtype=np.float32)
    paths = released_paths(args.targets)
    absent = sorted(set(groups) - set(paths))
    if absent:
        raise ValueError(f"No released broad ATAC track for groups: {', '.join(absent)}.")
    released = np.stack(
        [read_released_bins(paths[group], chromosome, bin_size, raw.shape[1]) for group in groups]
    )
    raw_matrix = raw.T
    released_matrix = released.T
    per_group = {
        group: {
            "pearson_r": pearson(raw[index], released[index]),
            "raw_mean": float(np.mean(raw[index], dtype=np.float64)),
            "released_mean": float(np.mean(released[index], dtype=np.float64)),
        }
        for index, group in enumerate(groups)
    }
    result = {
        "shard": str(args.shard.resolve()),
        "chromosome": chromosome,
        "bin_size": bin_size,
        "groups": groups,
        "coverage_spmr_vs_released_pearson_r": pearson(raw_matrix, released_matrix),
        "coverage_spmr_vs_released_double_centered_r": double_centered_pearson(
            raw_matrix, released_matrix
        ),
        "per_group": per_group,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
