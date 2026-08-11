#!/usr/bin/env python
"""Compute Liu HDMA whole-genome fragment totals for selected clusters."""

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
    fragment_totals_by_group,
    read_fragment_histogram,
)
from common import read_cell_metadata, selected_clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cluster-manifest", required=True, type=Path)
    parser.add_argument("--fragment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.cluster_manifest.read_text())
    groups = selected_clusters(manifest)
    selected = set(groups)
    metadata = read_cell_metadata(args.metadata)
    metadata = metadata[metadata["Cluster"].isin(selected)]
    totals = np.zeros(len(groups), dtype=np.float64)
    unmatched_histogram_cells = 0
    samples = sorted(metadata["sample"].unique())
    for index, sample in enumerate(samples, start=1):
        rows = metadata[metadata["sample"] == sample]
        cell_groups = dict(zip(rows["barcode"], rows["Cluster"], strict=True))
        histogram_path = args.fragment_root / sample / "statistics" / "histogram_cell.json"
        sample_totals, missing = fragment_totals_by_group(
            read_fragment_histogram(histogram_path),
            cell_groups,
            groups,
        )
        totals += sample_totals
        unmatched_histogram_cells += missing
        print(
            f"[{index}/{len(samples)}] {sample}, selected fragments={sample_totals.sum():,.0f}",
            flush=True,
        )
    if np.any(totals <= 0):
        empty = [groups[index] for index in np.flatnonzero(totals <= 0)]
        raise ValueError(f"Clusters with no fragments: {empty}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        groups=np.asarray(groups),
        total_fragments=totals,
        samples=np.asarray(samples),
    )
    summary = {
        "samples": len(samples),
        "clusters": len(groups),
        "selected_fragments": int(totals.sum()),
        "minimum_cluster_fragments": int(totals.min()),
        "maximum_cluster_fragments": int(totals.max()),
        "histogram_cells_without_selected_metadata": unmatched_histogram_cells,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
