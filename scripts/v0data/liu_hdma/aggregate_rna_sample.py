#!/usr/bin/env python
"""Aggregate one Liu HDMA sparse expression matrix by selected cell cluster."""

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
    aggregate_matrix_market_by_group,
    read_10x_features,
)
from common import read_cell_metadata, selected_clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cluster-manifest", required=True, type=Path)
    parser.add_argument("--expression-root", required=True, type=Path)
    parser.add_argument("--sample-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.cluster_manifest.read_text())
    groups = selected_clusters(manifest)
    selected = set(groups)
    metadata = read_cell_metadata(args.metadata)
    metadata = metadata[metadata["Cluster"].isin(selected)]
    samples = sorted(metadata["sample"].unique())
    if not 0 <= args.sample_index < len(samples):
        raise IndexError(f"Sample index {args.sample_index} is outside [0, {len(samples)}).")
    sample = samples[args.sample_index]
    rows = metadata[metadata["sample"] == sample]
    barcode_groups = dict(zip(rows["barcode"], rows["Cluster"], strict=True))
    sample_dir = args.expression_root / sample
    matrix_path = sample_dir / f"{sample}.matrix.mtx.gz"
    barcode_path = sample_dir / f"{sample}.barcodes.tsv.gz"
    feature_path = sample_dir / f"{sample}.features.tsv.gz"
    counts = aggregate_matrix_market_by_group(
        matrix_path,
        barcode_path,
        barcode_groups,
        groups,
    )
    gene_ids = read_10x_features(feature_path)
    if counts.shape != (len(groups), len(gene_ids)):
        raise ValueError(f"Unexpected aggregate shape {counts.shape}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{sample}.npz"
    np.savez_compressed(
        output,
        sample=np.asarray(sample),
        groups=np.asarray(groups),
        gene_ids=np.asarray(gene_ids),
        counts=counts,
        selected_cells=np.asarray(len(barcode_groups)),
    )
    print(f"Wrote {counts.shape} aggregate for {len(barcode_groups):,} cells to {output}.")


if __name__ == "__main__":
    main()
