#!/usr/bin/env python3
"""Prepare comparable raw-UMI gene-only RNA targets for all Zemke 2023 species."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.rna_tracks import (
    gene_supervision_exon_density_nonzero_means,
)
from alphagenome_ft.finetune.target_manifest import make_gene_only_config


def _rna_head(config: dict[str, Any]) -> dict[str, Any]:
    matches = [head for head in config.get("heads", ()) if head.get("kind") == "rna_seq"]
    if len(matches) != 1:
        raise ValueError(f"Expected one RNA head, found {len(matches)}.")
    return matches[0]


def prepare_gene_only_species(
    source: dict[str, Any],
    *,
    supervision_root: Path,
    output_dir: Path,
    correlation_loss_weight: float = 1.0,
) -> dict[str, Any]:
    """Write species manifests with pooled direct-gene exon-density scales."""
    result = copy.deepcopy(source)
    prepared: list[tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]] = []
    expected_groups: tuple[str, ...] | None = None
    for entry in result.get("species", ()):
        species = str(entry["name"])
        supervision_path = (
            supervision_root / species / "gene_expression_supervision.npz"
        ).resolve()
        source_targets = (supervision_root / species / "targets.json").resolve()
        if not supervision_path.exists() or not source_targets.exists():
            raise FileNotFoundError(f"Missing direct-gene artifacts for {species}.")
        config = make_gene_only_config(
            json.loads(source_targets.read_text()),
            head_id="zemke2023_rna",
            correlation_loss_weight=correlation_loss_weight,
        )
        groups, means, valid = gene_supervision_exon_density_nonzero_means(
            supervision_path
        )
        target_groups = tuple(str(target["label"]) for target in _rna_head(config)["targets"])
        if groups != target_groups:
            raise ValueError(f"RNA target order does not match gene groups for {species}.")
        if expected_groups is None:
            expected_groups = groups
        elif groups != expected_groups:
            raise ValueError("Zemke species use inconsistent direct-gene groups.")
        prepared.append((entry, config, means, valid))

    if not prepared:
        raise ValueError("Species configuration contains no entries.")
    mean_matrix = np.stack([item[2] for item in prepared])
    valid_matrix = np.stack([item[3] for item in prepared])
    pooled = np.asarray(
        [
            np.mean(mean_matrix[valid_matrix[:, index], index])
            if np.any(valid_matrix[:, index])
            else np.nan
            for index in range(mean_matrix.shape[1])
        ]
    )
    fallback = float(np.nanmedian(pooled))
    pooled = np.where(np.isfinite(pooled), pooled, fallback)
    if not np.all(np.isfinite(pooled)) or np.any(pooled <= 0):
        raise ValueError("Pooled exon-density scales must be finite and positive.")

    for entry, config, _, valid in prepared:
        species = str(entry["name"])
        rna = _rna_head(config)
        for target, mean in zip(rna["targets"], pooled, strict=True):
            target["nonzero_mean"] = float(mean)
        contract = config.setdefault("target_contract", {})
        contract["rna_gene_only"] = (
            "raw UMI counts summed by subclass, normalized to CPM, with pooled "
            "synthetic union-exon density scales"
        )
        contract["rna_coordinate_coverage_loss_weight"] = 0.0
        species_dir = output_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)
        destination = (species_dir / "targets.json").resolve()
        destination.write_text(json.dumps(config, indent=2) + "\n")
        entry["targets_config"] = str(destination)
        entry["direct_gene_valid_groups"] = int(valid.sum())

    result["rna_representation"] = (
        "raw UMI subclass CPM with direct gene-only supervision and pooled synthetic "
        "union-exon density scales"
    )
    result["rna_pooled_nonzero_means"] = pooled.tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "species.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--supervision-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--correlation-loss-weight", type=float, default=1.0)
    args = parser.parse_args()
    result = prepare_gene_only_species(
        json.loads(args.input.expanduser().resolve().read_text()),
        supervision_root=args.supervision_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        correlation_loss_weight=args.correlation_loss_weight,
    )
    print(
        f"Wrote {len(result['species'])} gene-only species manifests to "
        f"{args.output_dir.expanduser().resolve()}."
    )


if __name__ == "__main__":
    main()
