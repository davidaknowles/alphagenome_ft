#!/usr/bin/env python3
"""Copy a target manifest while retaining one requested head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.target_manifest import retain_target_heads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    result = retain_target_heads(json.loads(args.input.read_text()), (args.head,))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f'Retained head "{args.head}" in {args.output}.')


if __name__ == "__main__":
    main()
