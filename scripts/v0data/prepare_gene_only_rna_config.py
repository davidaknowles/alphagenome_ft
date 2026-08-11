#!/usr/bin/env python
"""Create a target manifest that supervises one RNA head only at gene level."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--correlation-loss-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.correlation_loss_weight < 0:
        raise ValueError("Correlation loss weight must be non-negative.")
    config = json.loads(args.input.read_text())
    config = copy.deepcopy(config)
    matches = [head for head in config.get("heads", ()) if head.get("id") == args.head]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one head named "{args.head}", found {len(matches)}.')
    head = matches[0]
    gene_supervision = head.get("gene_supervision")
    if not isinstance(gene_supervision, dict):
        raise ValueError(f'Head "{args.head}" does not define gene supervision.')
    gene_supervision["coverage_loss_weight"] = 0.0
    head["resolutions"] = [128]
    head["double_centered_correlation_loss_weight"] = args.correlation_loss_weight
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"Wrote gene-only {args.head} config with correlation weight "
        f"{args.correlation_loss_weight:g} to {args.output}."
    )


if __name__ == "__main__":
    main()
