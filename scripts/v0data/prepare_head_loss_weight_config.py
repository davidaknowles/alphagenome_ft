#!/usr/bin/env python3
"""Set one head's optimization weight in a copied target manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from alphagenome_ft.finetune import load_targets_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--weight", required=True, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.weight) or args.weight <= 0:
        raise ValueError("Head loss weight must be finite and positive.")
    config = load_targets_config(args.input)
    matches = [head for head in config.get("heads", ()) if head.get("id") == args.head]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one head named "{args.head}", found {len(matches)}.')
    matches[0]["loss_weight"] = args.weight
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f'Set head "{args.head}" loss weight to {args.weight:g} in {args.output}.')


if __name__ == "__main__":
    main()
