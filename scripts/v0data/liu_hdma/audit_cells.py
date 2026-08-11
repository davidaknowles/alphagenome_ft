#!/usr/bin/env python
"""Audit Liu HDMA cell assignments and define the paired target panel."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from common import read_cell_metadata, read_cluster_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--supplementary-table", required=True, type=Path)
    parser.add_argument("--bigwig-root", required=True, type=Path)
    parser.add_argument("--expression-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-barcode-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = read_cluster_specs(args.supplementary_table, args.bigwig_root)
    metadata = read_cell_metadata(args.metadata)
    annotated = {spec.cluster for spec in specs}
    unknown = sorted(set(metadata["Cluster"]) - annotated)
    if unknown:
        raise ValueError(f"Metadata clusters absent from Supplementary Table 2: {unknown}")

    sample_dirs = {path.name: path for path in args.expression_root.iterdir() if path.is_dir()}
    missing_samples = sorted(set(metadata["sample"]) - set(sample_dirs))
    if missing_samples:
        raise ValueError(f"Metadata samples without expression matrices: {missing_samples}")

    matched_barcodes = 0
    if not args.skip_barcode_check:
        for sample, rows in metadata.groupby("sample", sort=True):
            wanted = set(rows["barcode"])
            path = sample_dirs[sample] / f"{sample}.barcodes.tsv.gz"
            with gzip.open(path, "rt") as handle:
                found = sum(line.rstrip("\n") in wanted for line in handle)
            if found != len(wanted):
                raise ValueError(f"Only {found}/{len(wanted)} metadata cells occur in {path}.")
            matched_barcodes += found

    selected = [spec for spec in specs if spec.released_track]
    excluded = [spec for spec in specs if not spec.released_track]
    selected_ids = {spec.cluster for spec in selected}
    payload = {
        "dataset": "liu_hdma",
        "target_contract": (
            "Clusters retained in the published ChromBPNet panel; reconstructed targets use "
            "only measured fragments and raw expression counts."
        ),
        "metadata_cells": len(metadata),
        "metadata_samples": int(metadata["sample"].nunique()),
        "matched_rna_barcodes": matched_barcodes if not args.skip_barcode_check else None,
        "selected_cells": int(metadata["Cluster"].isin(selected_ids).sum()),
        "selected_clusters": [vars(spec) for spec in selected],
        "excluded_clusters": [vars(spec) for spec in excluded],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if not isinstance(value, list)}, indent=2
        )
    )
    print(f"Selected {len(selected)} clusters and excluded {len(excluded)} clusters.")


if __name__ == "__main__":
    main()
