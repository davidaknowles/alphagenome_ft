#!/usr/bin/env python
"""Audit RNA pseudobulk depth alongside ATAC depth-filter decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    return float(pd.Series(first).corr(pd.Series(second), method="spearman"))


def summarize_depths(
    groups: list[str],
    atac_fragments: np.ndarray,
    expression_depth: np.ndarray,
    cells: np.ndarray,
    retained_groups: set[str],
) -> dict[str, Any]:
    expected = (len(groups),)
    if any(values.shape != expected for values in (atac_fragments, expression_depth, cells)):
        raise ValueError("Every depth array must contain one value per group.")
    if any(np.any(values <= 0) for values in (atac_fragments, expression_depth, cells)):
        raise ValueError("Every group depth must be positive.")
    retained = np.asarray([group in retained_groups for group in groups])
    if not np.any(retained):
        raise ValueError("No groups are retained by the filtered target manifest.")
    quantiles = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {
        "num_groups": len(groups),
        "retained_groups": int(np.sum(retained)),
        "spearman_log_atac_vs_log_expression_depth": _spearman(
            np.log10(atac_fragments), np.log10(expression_depth)
        ),
        "spearman_log_atac_vs_log_cells": _spearman(
            np.log10(atac_fragments), np.log10(cells)
        ),
        "all_cell_quantiles": {
            str(q): float(np.quantile(cells, q)) for q in quantiles
        },
        "retained_cell_quantiles": {
            str(q): float(np.quantile(cells[retained], q)) for q in quantiles
        },
        "all_expression_depth_quantiles": {
            str(q): float(np.quantile(expression_depth, q)) for q in quantiles
        },
        "retained_expression_depth_quantiles": {
            str(q): float(np.quantile(expression_depth[retained], q)) for q in quantiles
        },
        "retained_below_500_cells": int(np.sum(cells[retained] < 500)),
        "retained_below_1000_cells": int(np.sum(cells[retained] < 1000)),
        "tracks": [
            {
                "group": group,
                "atac_fragments": int(atac),
                "expression_depth": float(expression),
                "cells": int(cell_count),
                "retained": bool(keep),
            }
            for group, atac, expression, cell_count, keep in zip(
                groups,
                atac_fragments,
                expression_depth,
                cells,
                retained,
                strict=True,
            )
        ],
    }


def _depth_map(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as source:
        return dict(
            zip(
                source["groups"].astype(str).tolist(),
                source["total_fragments"].astype(np.float64).tolist(),
                strict=True,
            )
        )


def _retained_groups(path: Path, head_id: str) -> set[str]:
    manifest = json.loads(path.read_text())
    head = next(head for head in manifest["heads"] if head["id"] == head_id)
    return {str(target["label"]) for target in head["targets"]}


def audit_liu(root: Path) -> dict[str, Any]:
    clusters = json.loads((root / "clusters.json").read_text())["selected_clusters"]
    groups = [str(record["cluster"]) for record in clusters]
    expression_depth = np.zeros((len(groups),), dtype=np.float64)
    for path in sorted((root / "rna_samples").glob("*.npz")):
        with np.load(path, allow_pickle=False) as sample:
            if sample["groups"].astype(str).tolist() != groups:
                raise ValueError(f"Group order differs in {path}.")
            expression_depth += sample["counts"].sum(axis=1, dtype=np.float64)
    metadata = pd.read_csv(
        "outputs/v0data/liu2026/source/per_cell_meta.csv",
        usecols=["Cluster"],
    )
    cell_counts = metadata["Cluster"].value_counts()
    cells = np.asarray([cell_counts.get(group, 0) for group in groups], dtype=np.float64)
    atac_by_group = _depth_map(root / "atac_totals.npz")
    atac = np.asarray([atac_by_group[group] for group in groups], dtype=np.float64)
    return summarize_depths(
        groups,
        atac,
        expression_depth,
        cells,
        _retained_groups(Path("outputs/v0data/liu-hdma-depth10m/targets.json"), "liu_atac"),
    )


def audit_johansen(
    species: str,
    rna_path: Path,
    atac_depth_path: Path,
    retained_manifest: Path,
) -> dict[str, Any]:
    with h5py.File(rna_path) as source:
        groups = _decode(source["obs/Group"][:])
        cells = np.asarray(source["obs/n_cells"][:], dtype=np.float64)
        expression_depth = np.asarray(source["X"][:], dtype=np.float64).sum(axis=1)
    atac_by_group = _depth_map(atac_depth_path)
    atac = np.asarray([atac_by_group[group] for group in groups], dtype=np.float64)
    return summarize_depths(
        groups,
        atac,
        expression_depth,
        cells,
        _retained_groups(retained_manifest, "allen_atac"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allen_root = Path(
        "/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/AllenBrainMultiome/RNA"
    )
    settings = {
        "human": Path("outputs/allen_atac_full_depth/shards/chr22_100bp.npz"),
        "macaque": Path(
            "outputs/v0data/johansen-fragment-atac/macaque/shards/NC_041754.1_100bp.npz"
        ),
        "marmoset": Path(
            "outputs/v0data/johansen-fragment-atac/marmoset/shards/chr1_100bp.npz"
        ),
    }
    audits = {"liu": audit_liu(Path("outputs/v0data/liu-hdma"))}
    for species, atac_depth_path in settings.items():
        title = species.capitalize()
        audits[f"johansen_{species}"] = audit_johansen(
            species,
            allen_root / f"{title}_HMBA_basalganglia_pseudobulk_aligned.h5ad",
            atac_depth_path,
            Path(f"outputs/v0data/johansen-fragment-joint-depth10m/{species}/targets.json"),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"audits": audits}, indent=2) + "\n")


if __name__ == "__main__":
    main()
