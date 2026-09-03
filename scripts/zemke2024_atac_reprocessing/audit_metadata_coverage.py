#!/usr/bin/env python
"""Audit metadata coverage of Zemke 2024 donor-level ATAC fragment resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.zemke2024_atac_reprocessing.aggregate import (
    discover_fragment_paths,
    read_fragment_histogram,
    read_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = read_metadata(args.metadata)
    donor_results: dict[str, dict[str, int | float]] = {}
    total_histogram_cells = 0
    total_metadata_cells = 0
    total_matched_cells = 0
    total_fragments = 0
    total_matched_fragments = 0
    missing_metadata: list[str] = []
    for path in discover_fragment_paths(args.fragment_root, max_donors=None):
        donor = path.parent.name
        groups = metadata.get(donor)
        if groups is None:
            missing_metadata.append(donor)
            continue
        histogram = read_fragment_histogram(path)
        matched = set(histogram).intersection(groups)
        matched_fragments = sum(histogram[cell] for cell in matched)
        donor_results[donor] = {
            "histogram_cells": len(histogram),
            "metadata_cells": len(groups),
            "matched_cells": len(matched),
            "matched_cell_fraction": len(matched) / len(histogram),
            "fragments": sum(histogram.values()),
            "matched_fragments": matched_fragments,
            "matched_fragment_fraction": matched_fragments / sum(histogram.values()),
        }
        total_histogram_cells += len(histogram)
        total_metadata_cells += len(groups)
        total_matched_cells += len(matched)
        total_fragments += sum(histogram.values())
        total_matched_fragments += matched_fragments
        print(
            f"{donor}: cells={len(matched):,}/{len(groups):,}, "
            f"fragments={matched_fragments:,}/{sum(histogram.values()):,}",
            flush=True,
        )
    result = {
        "fragment_donors": len(donor_results),
        "metadata_donors": len(metadata),
        "missing_metadata_for_fragment_donors": missing_metadata,
        "histogram_cells": total_histogram_cells,
        "metadata_cells": total_metadata_cells,
        "matched_cells": total_matched_cells,
        "matched_cell_fraction": total_matched_cells / total_histogram_cells,
        "fragments": total_fragments,
        "matched_fragments": total_matched_fragments,
        "matched_fragment_fraction": total_matched_fragments / total_fragments,
        "donors": donor_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
