#!/usr/bin/env python3
"""Set one head's objective weights in a copied target manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import load_targets_config


def set_head_objective(
    config: Mapping[str, Any],
    head_id: str,
    *,
    loss_weight: float | None = None,
    correlation_loss_weight: float | None = None,
) -> dict[str, Any]:
    """Copy a manifest and update objective weights for exactly one head."""
    if loss_weight is None and correlation_loss_weight is None:
        raise ValueError("At least one objective weight must be provided.")
    if loss_weight is not None and (not math.isfinite(loss_weight) or loss_weight <= 0):
        raise ValueError("Head loss weight must be finite and positive.")
    if correlation_loss_weight is not None and (
        not math.isfinite(correlation_loss_weight) or correlation_loss_weight < 0
    ):
        raise ValueError("Correlation loss weight must be finite and non-negative.")

    result = json.loads(json.dumps(config))
    matches = [head for head in result.get("heads", ()) if head.get("id") == head_id]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one head named "{head_id}", found {len(matches)}.')
    if loss_weight is not None:
        matches[0]["loss_weight"] = loss_weight
    if correlation_loss_weight is not None:
        matches[0]["double_centered_correlation_loss_weight"] = correlation_loss_weight
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--loss-weight", type=float)
    parser.add_argument("--correlation-loss-weight", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = set_head_objective(
        load_targets_config(args.input),
        args.head,
        loss_weight=args.loss_weight,
        correlation_loss_weight=args.correlation_loss_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f'Set objective weights for head "{args.head}" in {args.output}.')


if __name__ == "__main__":
    main()
