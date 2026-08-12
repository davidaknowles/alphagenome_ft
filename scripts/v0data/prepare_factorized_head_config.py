#!/usr/bin/env python3
"""Set one RNA output head to a low-rank projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.target_manifest import set_head_output_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--rank", required=True, type=int)
    args = parser.parse_args()
    result = set_head_output_rank(
        json.loads(args.input.read_text()),
        head_id=args.head,
        output_rank=args.rank,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote rank-{args.rank} output projection for {args.head} to {args.output}.")


if __name__ == "__main__":
    main()
