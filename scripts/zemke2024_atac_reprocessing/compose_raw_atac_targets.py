#!/usr/bin/env python
"""Replace the published Zemke 2024 ATAC head with a raw coverage head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _one_head(config: dict[str, Any], kind: str) -> dict[str, Any]:
    matches = [head for head in config.get("heads", ()) if head.get("kind") == kind]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {kind!r} head, found {len(matches)}.")
    return matches[0]


def compose_raw_atac_targets(
    gene_target_config: dict[str, Any], raw_atac_config: dict[str, Any]
) -> dict[str, Any]:
    """Preserve direct-gene RNA while substituting raw coverage-SPMR ATAC."""
    source_atac = _one_head(gene_target_config, "atac")
    raw_atac = _one_head(raw_atac_config, "atac")
    source_rna = _one_head(gene_target_config, "rna_seq")
    raw_targets = raw_atac.get("targets", ())
    if not raw_targets:
        raise ValueError("Raw ATAC head has no targets.")
    labels = [str(target.get("label", "")) for target in raw_targets]
    if len(set(labels)) != len(labels) or any(not label for label in labels):
        raise ValueError("Raw ATAC target labels must be unique and nonempty.")
    replacement = dict(raw_atac)
    replacement["id"] = source_atac["id"]
    result = dict(gene_target_config)
    result["heads"] = [replacement if head is source_atac else dict(head) for head in gene_target_config["heads"]]
    contract = dict(result.get("target_contract", {}))
    contract["atac"] = (
        "all metadata-matched quality-controlled fragments, accumulated as 100 bp "
        "coverage SPMR in 18 broad subclasses"
    )
    contract["atac_replaced_published_channels"] = len(source_atac.get("targets", ()))
    contract["atac_raw_channels"] = len(raw_targets)
    contract["rna_head_preserved"] = source_rna["id"]
    result["target_contract"] = contract
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-targets", type=Path, required=True)
    parser.add_argument("--raw-atac-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compose_raw_atac_targets(
        json.loads(args.gene_targets.read_text()), json.loads(args.raw_atac_targets.read_text())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
