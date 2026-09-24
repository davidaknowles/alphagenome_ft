#!/usr/bin/env python
"""Combine completed per-model XQTL benchmark score files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.input_root / f"models_{name}" / f"scores_{name}.tsv" for name in ("head_only", "lora", "locon")]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing completed model score files: " + ", ".join(map(str, missing)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = None
    total = 0
    with args.output.open("w", newline="") as output:
        writer = None
        for path in paths:
            with path.open() as source:
                reader = csv.DictReader(source, delimiter="\t")
                if header is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(output, fieldnames=header, delimiter="\t")
                    writer.writeheader()
                elif reader.fieldnames != header:
                    raise ValueError(f"Incompatible columns in {path}")
                for row in reader:
                    writer.writerow(row)
                    total += 1
    print(f"wrote {total} model score rows to {args.output}")


if __name__ == "__main__":
    main()
