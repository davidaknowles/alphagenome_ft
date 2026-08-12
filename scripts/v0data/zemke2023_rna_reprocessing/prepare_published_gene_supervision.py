#!/usr/bin/env python3
"""Build direct-gene supervision by integrating published Zemke RNA tracks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from alphagenome_ft.finetune.reprocessing import normalize_counts_per_million
from scripts.v0data.zemke2023_rna_reprocessing.audit_gene_track_agreement import (
    _integrate_track,
)


def _rna_head(config: dict[str, Any]) -> dict[str, Any]:
    heads = [head for head in config.get("heads", ()) if head.get("kind") == "rna_seq"]
    if len(heads) != 1:
        raise ValueError(f"Expected one RNA head, found {len(heads)}.")
    return heads[0]


def prepare_published_gene_supervision(
    *,
    source_supervision: Path,
    source_targets: Path,
    output_dir: Path,
    species: str,
    gene_loss_weight: float = 1.0,
    coverage_loss_weight: float = 1.0,
    correlation_loss_weight: float | None = 10.0,
) -> dict[str, Any]:
    """Integrate published tracks over source union-exon geometry and write CPM targets."""
    for name, value in (
        ("gene_loss_weight", gene_loss_weight),
        ("coverage_loss_weight", coverage_loss_weight),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if correlation_loss_weight is not None and (
        not np.isfinite(correlation_loss_weight) or correlation_loss_weight < 0
    ):
        raise ValueError("correlation_loss_weight must be finite and non-negative.")

    config = json.loads(source_targets.read_text())
    rna_head = _rna_head(config)
    target_groups = tuple(str(target["label"]) for target in rna_head["targets"])
    with np.load(source_supervision, allow_pickle=False) as source:
        groups = tuple(str(value) for value in source["groups"])
        if groups != target_groups:
            raise ValueError("Source gene groups do not match published RNA target order.")
        payload = {key: np.asarray(source[key]) for key in source.files if key != "cpm"}
        chromosomes = source["chromosomes"].astype(str)
        exon_offsets = np.asarray(source["exon_offsets"], dtype=np.int64)
        exon_starts = np.asarray(source["exon_starts"], dtype=np.int64)
        exon_ends = np.asarray(source["exon_ends"], dtype=np.int64)
    gene_indices = np.arange(len(chromosomes), dtype=np.int64)
    integrated = np.stack(
        [
            _integrate_track(
                Path(target["path"]),
                chromosomes=chromosomes,
                exon_offsets=exon_offsets,
                exon_starts=exon_starts,
                exon_ends=exon_ends,
                indices=gene_indices,
            )
            for target in rna_head["targets"]
        ],
        axis=0,
    )
    cpm = normalize_counts_per_million(integrated)
    payload["cpm"] = cpm
    payload["group_valid"] = np.ones((len(groups),), dtype=bool)

    output_dir.mkdir(parents=True, exist_ok=True)
    supervision_path = output_dir / "gene_expression_supervision.npz"
    np.savez_compressed(supervision_path, **payload)
    output_config = copy.deepcopy(config)
    output_rna = _rna_head(output_config)
    output_rna["gene_supervision"] = {
        "path": str(supervision_path.resolve()),
        "loss_weight": gene_loss_weight,
        "coverage_loss_weight": coverage_loss_weight,
    }
    if correlation_loss_weight is not None:
        output_rna["double_centered_correlation_loss_weight"] = correlation_loss_weight
    contract = output_config.setdefault("target_contract", {})
    contract["rna_gene"] = (
        "published RPKM integrated over union exons, then normalized to counts per million"
    )
    output_targets = output_dir / "targets.json"
    output_targets.write_text(json.dumps(output_config, indent=2) + "\n")

    manifest = {
        "species": species,
        "source_supervision": str(source_supervision.resolve()),
        "source_targets": str(source_targets.resolve()),
        "groups": len(groups),
        "genes": len(chromosomes),
        "normalization": contract["rna_gene"],
        "gene_loss_weight": gene_loss_weight,
        "coverage_loss_weight": coverage_loss_weight,
        "double_centered_correlation_loss_weight": correlation_loss_weight,
        "minimum_group_total_before_normalization": float(integrated.sum(axis=1).min()),
        "maximum_group_total_before_normalization": float(integrated.sum(axis=1).max()),
        "targets": str(output_targets.resolve()),
        "gene_supervision": str(supervision_path.resolve()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-supervision", required=True, type=Path)
    parser.add_argument("--source-targets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--species", required=True)
    parser.add_argument("--gene-loss-weight", type=float, default=1.0)
    parser.add_argument("--coverage-loss-weight", type=float, default=1.0)
    parser.add_argument("--correlation-loss-weight", type=float, default=10.0)
    args = parser.parse_args()
    manifest = prepare_published_gene_supervision(
        source_supervision=args.source_supervision.expanduser().resolve(),
        source_targets=args.source_targets.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        species=args.species,
        gene_loss_weight=args.gene_loss_weight,
        coverage_loss_weight=args.coverage_loss_weight,
        correlation_loss_weight=args.correlation_loss_weight,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
