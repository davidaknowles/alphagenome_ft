#!/usr/bin/env python
"""Aggregate Zemke 2024 donor fragments into broad-subclass ATAC targets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reprocessing import (
    BinnedAtacAccumulator,
    fragment_totals_by_group,
    stream_tabix_fragments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--chromosome-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=100)
    parser.add_argument("--max-donors", type=int)
    parser.add_argument("--tabix", default="tabix")
    return parser.parse_args()


def discover_fragment_paths(root: Path, max_donors: int | None) -> list[Path]:
    paths = sorted(
        path
        for path in root.glob("*/*.tsv.gz")
        if Path(f"{path}.tbi").exists()
    )
    if max_donors is not None:
        paths = paths[:max_donors]
    if not paths:
        raise FileNotFoundError(f"No tabix-indexed fragment files found under {root}.")
    return paths


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Return donor-local barcode to published broad-subclass labels."""
    groups_by_donor: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"bacrode", "orig.ident", "subclass"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"Expected {sorted(required)} columns in {path}.")
        for row in reader:
            donor = row["orig.ident"].strip()
            barcode = row["bacrode"].strip()
            group = row["subclass"].strip()
            prefix = f"{donor}_"
            if not donor or not barcode.startswith(prefix) or not group:
                continue
            local_barcode = barcode[len(prefix) :]
            previous = groups_by_donor.setdefault(donor, {}).setdefault(local_barcode, group)
            if previous != group:
                raise ValueError(
                    f"Barcode {local_barcode!r} maps to multiple groups in donor {donor!r}."
                )
    if not groups_by_donor:
        raise ValueError(f"No donor-local barcodes found in {path}.")
    return groups_by_donor


def read_fragment_histogram(path: Path) -> dict[str, int]:
    histogram_path = path.parent / "statistics" / "histogram_cell.json"
    payload = json.loads(histogram_path.read_text())
    values = payload.get("values")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"Missing per-cell fragment counts in {histogram_path}.")
    return {str(cell): int(count) for cell, count in values.items()}


def main() -> None:
    args = parse_args()
    paths = discover_fragment_paths(args.fragment_root, args.max_donors)
    groups_by_donor = read_metadata(args.metadata)
    histograms = {path: read_fragment_histogram(path) for path in paths}
    cell_groups_by_path: dict[Path, dict[str, str]] = {}
    unmatched_donors: list[str] = []
    for path in paths:
        donor = path.parent.name
        cell_groups = groups_by_donor.get(donor)
        if cell_groups is None:
            unmatched_donors.append(donor)
            continue
        cell_groups_by_path[path] = cell_groups
    if unmatched_donors:
        raise ValueError(f"No metadata for fragment donors: {', '.join(unmatched_donors)}.")

    all_groups = sorted({group for values in cell_groups_by_path.values() for group in values.values()})
    total_fragments = np.zeros(len(all_groups), dtype=np.float64)
    group_indices = {group: index for index, group in enumerate(all_groups)}
    histogram_missing_cells = 0
    for path, histogram in histograms.items():
        totals, missing = fragment_totals_by_group(histogram, cell_groups_by_path[path], all_groups)
        total_fragments += totals
        histogram_missing_cells += missing
    retained = np.flatnonzero(total_fragments > 0)
    groups = [all_groups[index] for index in retained]
    total_fragments = total_fragments[retained]
    group_indices = {group: index for index, group in enumerate(groups)}

    accumulator = BinnedAtacAccumulator(
        num_groups=len(groups),
        chromosome_size=args.chromosome_size,
        bin_size=args.bin_size,
    )
    matched_records = 0
    missing_records = 0
    for index, path in enumerate(paths, start=1):
        cell_group_indices = {
            cell: group_indices[group]
            for cell, group in cell_groups_by_path[path].items()
            if group in group_indices
        }
        matched, missing = stream_tabix_fragments(
            path,
            args.chromosome,
            cell_group_indices=cell_group_indices,
            accumulator=accumulator,
            tabix=args.tabix,
        )
        matched_records += matched
        missing_records += missing
        print(f"[{index}/{len(paths)}] {path.parent.name}, matched={matched:,}, missing={missing:,}", flush=True)

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
    metadata_cells = sum(len(groups) for groups in cell_groups_by_path.values())
    summary = {
        "output": str(args.output.resolve()),
        "chromosome": args.chromosome,
        "chromosome_size": args.chromosome_size,
        "bin_size": args.bin_size,
        "groups": groups,
        "fragment_files": len(paths),
        "metadata_cells": metadata_cells,
        "histogram_cells_missing_metadata": histogram_missing_cells,
        "chromosome_records_matched": matched_records,
        "chromosome_records_missing_metadata": missing_records,
        "whole_genome_fragments": int(total_fragments.sum()),
        "tn5_shift": True,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
