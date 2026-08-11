#!/usr/bin/env python
"""Aggregate Allen HMBA fragment resources into normalized binned ATAC targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reprocessing import (
    BinnedAtacAccumulator,
    fragment_totals_by_group,
    match_fragment_library,
    read_cell_groups_by_library,
    read_cell_groups,
)


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
    parser.add_argument("--no-tn5-shift", action="store_true")
    parser.add_argument(
        "--fragment-cell-id-mode",
        choices=("direct", "library_barcode"),
        default="direct",
    )
    return parser.parse_args()


def discover_fragment_paths(
    root: Path,
    max_files: int | None,
    library_suffix: str | None = None,
) -> list[Path]:
    paths = sorted(
        path
        for path in root.glob("*/*.gz")
        if not path.name.endswith(".tbi")
        and Path(f"{path}.tbi").exists()
        and (library_suffix is None or path.parent.name.endswith(library_suffix))
    )
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No tabix-indexed fragment files found under {root}.")
    return paths


def read_fragment_histogram(path: Path) -> dict[str, int]:
    histogram_path = path.parent / "statistics" / "histogram_cell.json"
    payload = json.loads(histogram_path.read_text())
    values = payload.get("values")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"Missing per-cell fragment counts in {histogram_path}.")
    return {str(cell): int(count) for cell, count in values.items()}


def add_chunk(
    accumulator: BinnedAtacAccumulator,
    group_indices: list[int],
    starts: list[int],
    ends: list[int],
    counts: list[int],
) -> None:
    accumulator.add(group_indices, starts, ends, counts)
    group_indices.clear()
    starts.clear()
    ends.clear()
    counts.clear()


def stream_chromosome(
    path: Path,
    chromosome: str,
    *,
    tabix: str,
    cell_group_indices: dict[str, int],
    accumulator: BinnedAtacAccumulator,
    chunk_size: int,
) -> tuple[int, int]:
    process = subprocess.Popen(
        [tabix, str(path), chromosome],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    group_indices: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    counts: list[int] = []
    matched = 0
    missing = 0
    for line in process.stdout:
        fields = line.rstrip().split("\t")
        if len(fields) < 5:
            process.kill()
            raise ValueError(f"Malformed fragment row in {path}, {line[:200]!r}")
        group_index = cell_group_indices.get(fields[3])
        if group_index is None:
            missing += 1
            continue
        group_indices.append(group_index)
        starts.append(int(fields[1]))
        ends.append(int(fields[2]))
        counts.append(int(fields[4]))
        matched += 1
        if len(group_indices) >= chunk_size:
            add_chunk(accumulator, group_indices, starts, ends, counts)
    if group_indices:
        add_chunk(accumulator, group_indices, starts, ends, counts)
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"tabix failed for {path} with code {return_code}, {stderr.strip()}")
    return matched, missing


def main() -> None:
    args = parse_args()
    fragment_paths = discover_fragment_paths(
        args.fragment_root,
        args.max_files,
        args.fragment_library_suffix,
    )
    histograms = {path: read_fragment_histogram(path) for path in fragment_paths}
    fragment_cell_groups = {}
    fragment_libraries = {}
    if args.fragment_cell_id_mode == "library_barcode":
        cell_groups_by_library = read_cell_groups_by_library(args.cell_metadata_h5ad)
        for path, histogram in histograms.items():
            library, cell_groups = match_fragment_library(
                histogram,
                cell_groups_by_library,
            )
            fragment_cell_groups[path] = cell_groups
            fragment_libraries[path.parent.name] = library
        metadata_cells = sum(len(groups) for groups in cell_groups_by_library.values())
    else:
        cell_groups = read_cell_groups(args.cell_metadata_h5ad)
        fragment_cell_groups = {path: cell_groups for path in fragment_paths}
        metadata_cells = len(cell_groups)
    all_groups = sorted(
        {group for cell_groups in fragment_cell_groups.values() for group in cell_groups.values()}
    )

    all_total_fragments = np.zeros(len(all_groups), dtype=np.float64)
    histogram_missing_cells = 0
    for path, histogram in histograms.items():
        file_totals, missing = fragment_totals_by_group(
            histogram, fragment_cell_groups[path], all_groups
        )
        all_total_fragments += file_totals
        histogram_missing_cells += missing
    retained = np.flatnonzero(all_total_fragments > 0)
    groups = [all_groups[index] for index in retained]
    total_fragments = all_total_fragments[retained]
    group_indices = {group: index for index, group in enumerate(groups)}

    accumulator = BinnedAtacAccumulator(
        num_groups=len(groups),
        chromosome_size=args.chromosome_size,
        bin_size=args.bin_size,
        tn5_shift=not args.no_tn5_shift,
    )
    matched_records = 0
    missing_records = 0
    for index, path in enumerate(fragment_paths, start=1):
        cell_group_indices = {
            cell: group_indices[group]
            for cell, group in fragment_cell_groups[path].items()
            if group in group_indices
        }
        matched, missing = stream_chromosome(
            path,
            args.chromosome,
            tabix=args.tabix,
            cell_group_indices=cell_group_indices,
            accumulator=accumulator,
            chunk_size=args.chunk_size,
        )
        matched_records += matched
        missing_records += missing
        print(
            f"[{index}/{len(fragment_paths)}] {path.parent.name}, "
            f"matched={matched:,}, missing={missing:,}",
            flush=True,
        )

    insertion_spmr, coverage_spmr = accumulator.normalized(total_fragments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        groups=np.asarray(groups),
        chromosome=np.asarray(args.chromosome),
        chromosome_size=np.asarray(args.chromosome_size),
        bin_size=np.asarray(args.bin_size),
        insertion_spmr=insertion_spmr,
        coverage_spmr=coverage_spmr,
        total_fragments=total_fragments,
        chromosome_fragments=accumulator.fragment_counts,
    )
    summary = {
        "output": str(args.output.resolve()),
        "chromosome": args.chromosome,
        "chromosome_size": args.chromosome_size,
        "bin_size": args.bin_size,
        "groups": len(groups),
        "fragment_files": len(fragment_paths),
        "metadata_cells": metadata_cells,
        "fragment_cell_id_mode": args.fragment_cell_id_mode,
        "fragment_libraries": fragment_libraries,
        "histogram_cells_missing_metadata": histogram_missing_cells,
        "chromosome_records_matched": matched_records,
        "chromosome_records_missing_metadata": missing_records,
        "whole_genome_fragments": int(total_fragments.sum()),
        "tn5_shift": not args.no_tn5_shift,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
