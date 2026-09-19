#!/usr/bin/env python
"""Combine per-model eQTL score files from concurrent benchmark jobs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.output_dir.glob("scores_*.tsv"))
    if not paths:
        raise FileNotFoundError(f"No per-model score files found in {args.output_dir}")
    destination = args.output_dir / "scores.tsv"
    header = None
    rows = 0
    with destination.open("w") as output:
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
                    rows += 1
    print(f"wrote {rows} rows to {destination}")


if __name__ == "__main__":
    main()
