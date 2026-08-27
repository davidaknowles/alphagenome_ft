#!/usr/bin/env python3
"""Prepare a gene-only Zemke 2024 target with direct-count output scales."""

from __future__ import annotations

import argparse
import json
import math
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


def prepare_gene_only_target(
    source: dict[str, Any],
    *,
    supervision_path: Path,
    correlation_loss_weight: float = 1.0,
) -> dict[str, Any]:
    """Disable coordinate RNA loss and derive channel scales from raw UMI CPM."""
    supervision_path = supervision_path.expanduser().resolve()
    if not supervision_path.exists():
        raise FileNotFoundError(supervision_path)
    result = make_gene_only_config(
        source,
        head_id="zemke2024_all_rna",
        correlation_loss_weight=correlation_loss_weight,
        gene_supervision_path=str(supervision_path),
    )
    groups, means, valid = gene_supervision_exon_density_nonzero_means(supervision_path)
    targets = _rna_head(result)["targets"]
    target_groups = tuple(str(target["label"]) for target in targets)
    if groups != target_groups:
        raise ValueError("RNA target order does not match direct-gene groups.")
    valid_means = means[valid]
    if not len(valid_means) or not np.all(np.isfinite(valid_means)) or np.any(valid_means <= 0):
        raise ValueError("Valid direct-gene output scales must be finite and positive.")
    fallback = float(np.median(valid_means))
    scales = np.where(valid, means, fallback)
    for target, scale in zip(targets, scales, strict=True):
        target["nonzero_mean"] = float(scale)
    contract = result.setdefault("target_contract", {})
    contract["rna_gene_only"] = (
        "raw UMI counts summed by released broad subclass, normalized to CPM, with "
        "synthetic union-exon density scales; four unreleased subtype channels are masked"
    )
    contract["rna_coordinate_coverage_loss_weight"] = 0.0
    contract["rna_masked_direct_gene_groups"] = [
        group for group, is_valid in zip(groups, valid, strict=True) if not is_valid
    ]
    contract["rna_invalid_scale_fallback"] = fallback
    if not math.isfinite(fallback) or fallback <= 0:
        raise ValueError("Fallback direct-gene output scale must be finite and positive.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--supervision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--correlation-loss-weight", type=float, default=1.0)
    args = parser.parse_args()
    result = prepare_gene_only_target(
        json.loads(args.input.expanduser().resolve().read_text()),
        supervision_path=args.supervision,
        correlation_loss_weight=args.correlation_loss_weight,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote gene-only Zemke 2024 target manifest to {output}.")


if __name__ == "__main__":
    main()
