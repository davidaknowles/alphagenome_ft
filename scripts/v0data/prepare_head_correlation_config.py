#!/usr/bin/env python3
"""Set one head's gene-correlation objective weights in a copied manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from alphagenome_ft.finetune import load_targets_config


def prepare_config(
    input_path: Path,
    *,
    head_id: str,
    double_centered_weight: float,
    row_centered_weight: float,
) -> dict[str, object]:
    """Copy a target config and set both gene-correlation weights."""
    for label, weight in (
        ("double-centered", double_centered_weight),
        ("row-centered", row_centered_weight),
    ):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{label} correlation weight must be finite and non-negative.")
    config = load_targets_config(input_path)
    matches = [head for head in config.get("heads", ()) if head.get("id") == head_id]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one head named "{head_id}", found {len(matches)}.')
    head = matches[0]
    if "gene_supervision" not in head:
        raise ValueError(f'Head "{head_id}" has no direct gene supervision.')
    head["double_centered_correlation_loss_weight"] = double_centered_weight
    head["row_centered_correlation_loss_weight"] = row_centered_weight
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--double-centered-weight", type=float, default=0.0)
    parser.add_argument("--row-centered-weight", type=float, default=0.0)
    args = parser.parse_args()
    config = prepare_config(
        args.input,
        head_id=args.head,
        double_centered_weight=args.double_centered_weight,
        row_centered_weight=args.row_centered_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(
        f'Set head "{args.head}" double-centered weight to '
        f"{args.double_centered_weight:g} and row-centered weight to "
        f"{args.row_centered_weight:g} in {args.output}."
    )


if __name__ == "__main__":
    main()
