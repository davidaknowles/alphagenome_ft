#!/usr/bin/env python
"""Combine Liu HDMA sample aggregates and write cluster-level CPM expression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from common import read_cell_metadata, selected_clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cluster-manifest", required=True, type=Path)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.cluster_manifest.read_text())
    groups = selected_clusters(manifest)
    metadata = read_cell_metadata(args.metadata)
    selected = set(groups)
    samples = sorted(metadata.loc[metadata["Cluster"].isin(selected), "sample"].unique())
    paths = [args.sample_dir / f"{sample}.npz" for sample in samples]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} RNA sample aggregates: {missing[:5]}")

    total_counts: np.ndarray | None = None
    gene_ids: tuple[str, ...] | None = None
    selected_cells = 0
    for path in paths:
        with np.load(path) as sample:
            sample_groups = tuple(sample["groups"].astype(str))
            sample_genes = tuple(sample["gene_ids"].astype(str))
            if sample_groups != groups:
                raise ValueError(f"Group order differs in {path}.")
            if gene_ids is None:
                gene_ids = sample_genes
                total_counts = np.zeros(sample["counts"].shape, dtype=np.float64)
            elif sample_genes != gene_ids:
                raise ValueError(f"Feature order differs in {path}.")
            total_counts += sample["counts"]
            selected_cells += int(sample["selected_cells"])
    assert total_counts is not None and gene_ids is not None
    library_sizes = total_counts.sum(axis=1, dtype=np.float64)
    if np.any(library_sizes <= 0):
        empty = [groups[index] for index in np.flatnonzero(library_sizes <= 0)]
        raise ValueError(f"Clusters with no RNA counts: {empty}")
    cpm = (total_counts / library_sizes[:, None] * 1_000_000.0).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as output:
        output.create_dataset("X", data=cpm, compression="gzip", shuffle=True)
        obs = output.create_group("obs")
        obs.create_dataset("Group", data=np.asarray(groups, dtype=object), dtype=string_dtype)
        var = output.create_group("var")
        var.create_dataset("gene_id", data=np.asarray(gene_ids, dtype=object), dtype=string_dtype)
        output.attrs["normalization"] = "cluster pseudobulk counts per million"
        output.attrs["selected_cells"] = selected_cells
    summary = {
        "samples": len(paths),
        "clusters": len(groups),
        "genes": len(gene_ids),
        "selected_cells": selected_cells,
        "minimum_library_size": float(library_sizes.min()),
        "maximum_library_size": float(library_sizes.max()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
