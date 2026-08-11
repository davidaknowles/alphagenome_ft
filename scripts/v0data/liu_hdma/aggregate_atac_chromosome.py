#!/usr/bin/env python
"""Build one chromosome of Liu HDMA cluster-level ATAC coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reprocessing import (
    BinnedAtacAccumulator,
    stream_tabix_fragments,
)
from common import read_cell_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--totals", required=True, type=Path)
    parser.add_argument("--fragment-root", required=True, type=Path)
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--chromosome-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bin-size", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--tabix", default="tabix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.totals) as totals_file:
        groups = tuple(totals_file["groups"].astype(str))
        total_fragments = totals_file["total_fragments"].astype(np.float64)
        samples = tuple(totals_file["samples"].astype(str))
    group_indices = {group: index for index, group in enumerate(groups)}
    metadata = read_cell_metadata(args.metadata)
    metadata = metadata[metadata["sample"].isin(samples)]
    accumulator = BinnedAtacAccumulator(
        num_groups=len(groups),
        chromosome_size=args.chromosome_size,
        bin_size=args.bin_size,
        tn5_shift=False,
    )
    matched_records = 0
    missing_records = 0
    for index, sample in enumerate(samples, start=1):
        rows = metadata[metadata["sample"] == sample]
        cell_group_indices = {
            barcode: group_indices[group]
            for barcode, group in zip(rows["barcode"], rows["Cluster"], strict=True)
            if group in group_indices
        }
        fragment_path = args.fragment_root / sample / f"{sample}.fragments.tsv.gz"
        matched, missing = stream_tabix_fragments(
            fragment_path,
            args.chromosome,
            cell_group_indices=cell_group_indices,
            accumulator=accumulator,
            chunk_size=args.chunk_size,
            tabix=args.tabix,
        )
        matched_records += matched
        missing_records += missing
        print(
            f"[{index}/{len(samples)}] {sample}, matched={matched:,}, unselected={missing:,}",
            flush=True,
        )
    _, coverage_spmr = accumulator.normalized(total_fragments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        groups=np.asarray(groups),
        chromosome=np.asarray(args.chromosome),
        chromosome_size=np.asarray(args.chromosome_size),
        bin_size=np.asarray(args.bin_size),
        coverage_spmr=coverage_spmr,
        total_fragments=total_fragments,
        chromosome_fragments=accumulator.fragment_counts,
    )
    summary = {
        "chromosome": args.chromosome,
        "chromosome_size": args.chromosome_size,
        "clusters": len(groups),
        "samples": len(samples),
        "matched_records": matched_records,
        "unselected_records": missing_records,
        "normalization": "mean fragment coverage per 100 bp bin per million selected fragments",
        "downsampling": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
