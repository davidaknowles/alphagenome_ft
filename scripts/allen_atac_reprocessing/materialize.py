#!/usr/bin/env python
"""Materialize chromosome ATAC shards as one BigWig per cell group."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import pyBigWig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--fai", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--field", default="coverage_spmr")
    parser.add_argument("--head-id", default="allen_atac")
    parser.add_argument("--chunk-entries", type=int, default=250_000)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def read_fai(path: Path) -> list[tuple[str, int]]:
    entries = []
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) >= 2:
                entries.append((fields[0], int(fields[1])))
    if not entries:
        raise ValueError(f"No chromosome sizes found in {path}.")
    return entries


def discover_shards(shard_dir: Path) -> dict[str, Path]:
    result = {}
    for path in shard_dir.glob("*.npz"):
        with np.load(path) as shard:
            chromosome = str(shard["chromosome"])
        if chromosome in result:
            raise ValueError(f"Multiple shards found for {chromosome}.")
        result[chromosome] = path
    if not result:
        raise FileNotFoundError(f"No chromosome shards found in {shard_dir}.")
    return result


def add_values(
    output: pyBigWig.pyBigWig,
    chromosome: str,
    values: np.ndarray,
    chromosome_size: int,
    bin_size: int,
    chunk_entries: int,
) -> tuple[float, int]:
    nonzero = np.flatnonzero(np.isfinite(values) & (values > 0))
    weighted_sum = 0.0
    covered_bases = 0
    for offset in range(0, len(nonzero), chunk_entries):
        indices = nonzero[offset : offset + chunk_entries]
        starts = indices * bin_size
        ends = np.minimum(starts + bin_size, chromosome_size)
        chunk_values = values[indices].astype(np.float64, copy=False)
        widths = ends - starts
        output.addEntries(
            [chromosome] * len(indices),
            starts.tolist(),
            ends=ends.tolist(),
            values=chunk_values.tolist(),
        )
        weighted_sum += float(np.sum(chunk_values * widths, dtype=np.float64))
        covered_bases += int(np.sum(widths, dtype=np.int64))
    return weighted_sum, covered_bases


def main() -> None:
    args = parse_args()
    chromosome_sizes = read_fai(args.fai)
    size_by_chromosome = dict(chromosome_sizes)
    shards = discover_shards(args.shard_dir)
    unknown = sorted(set(shards) - set(size_by_chromosome))
    if unknown:
        raise ValueError(f"Shard chromosomes are absent from the FASTA index, {unknown}.")

    first_path = next(iter(shards.values()))
    with np.load(first_path) as first:
        groups = first["groups"].astype(str).tolist()
    filenames = [f"{safe_name(group)}.bw" for group in groups]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Sanitized group names are not unique.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [args.output_dir / filename for filename in filenames]
    existing = [path for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace {len(existing)} existing BigWig file(s).")
    temporary_paths = [path.with_name(f"{path.name}.tmp-{os.getpid()}") for path in final_paths]
    outputs = [pyBigWig.open(str(path), "w") for path in temporary_paths]
    nonzero_sums = np.zeros(len(groups), dtype=np.float64)
    nonzero_bases = np.zeros(len(groups), dtype=np.int64)
    try:
        for output in outputs:
            output.addHeader(chromosome_sizes)
        for chromosome, chromosome_size in chromosome_sizes:
            path = shards.get(chromosome)
            if path is None:
                continue
            with np.load(path) as shard:
                shard_groups = shard["groups"].astype(str).tolist()
                if shard_groups != groups:
                    raise ValueError(f"Group order differs in {path}.")
                if int(shard["chromosome_size"]) != chromosome_size:
                    raise ValueError(f"Chromosome size differs in {path}.")
                bin_size = int(shard["bin_size"])
                matrix = shard[args.field]
            for group_index, output in enumerate(outputs):
                value_sum, base_count = add_values(
                    output,
                    chromosome,
                    matrix[group_index],
                    chromosome_size,
                    bin_size,
                    args.chunk_entries,
                )
                nonzero_sums[group_index] += value_sum
                nonzero_bases[group_index] += base_count
            print(f"Materialized {chromosome} from {path.name}.", flush=True)
    finally:
        for output in outputs:
            output.close()

    for temporary, final in zip(temporary_paths, final_paths, strict=True):
        temporary.replace(final)
    targets = []
    for index, (group, path) in enumerate(zip(groups, final_paths, strict=True)):
        targets.append(
            {
                "path": str(path.resolve()),
                "label": group,
                "strand": ".",
                "nonzero_mean": float(nonzero_sums[index] / nonzero_bases[index]),
            }
        )
    config = {
        "heads": [
            {
                "id": args.head_id,
                "source": "predefined",
                "kind": "atac",
                "resolutions": [1, 128],
                "apply_squashing": False,
                "targets": targets,
            }
        ]
    }
    (args.output_dir / "targets.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(final_paths)} BigWigs and targets.json to {args.output_dir}.")


if __name__ == "__main__":
    main()
