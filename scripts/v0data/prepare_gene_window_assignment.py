#!/usr/bin/env python3
"""Set direct-gene window assignment in a copied target manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import load_targets_config
from alphagenome_ft.finetune.target_manifest import set_gene_window_assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument(
        "--assignment",
        choices=("full_span", "max_exon_overlap_scaled"),
        required=True,
    )
    args = parser.parse_args()
    config = set_gene_window_assignment(
        load_targets_config(args.input),
        head_id=args.head,
        assignment=args.assignment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f'Set head "{args.head}" window assignment to {args.assignment}.')


if __name__ == "__main__":
    main()
