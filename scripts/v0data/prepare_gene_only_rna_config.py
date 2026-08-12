#!/usr/bin/env python
"""Create a target manifest that supervises one RNA head only at gene level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.target_manifest import make_gene_only_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--correlation-loss-weight", type=float, default=0.0)
    parser.add_argument("--row-correlation-loss-weight", type=float, default=0.0)
    parser.add_argument("--gene-supervision-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.input.read_text())
    config = make_gene_only_config(
        config,
        head_id=args.head,
        correlation_loss_weight=args.correlation_loss_weight,
        row_correlation_loss_weight=args.row_correlation_loss_weight,
        gene_supervision_path=(
            str(args.gene_supervision_path.expanduser().resolve())
            if args.gene_supervision_path is not None
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"Wrote gene-only {args.head} config with double-centered correlation weight "
        f"{args.correlation_loss_weight:g} and row-centered correlation weight "
        f"{args.row_correlation_loss_weight:g} to {args.output}."
    )


if __name__ == "__main__":
    main()
