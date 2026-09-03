#!/usr/bin/env python
"""Collate species-specific Johansen ATAC split-half reliability estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_FIELDS = (
    "species",
    "chromosome",
    "signal",
    "groups",
    "split_half_double_centered_r",
    "full_target_reliability_estimate",
    "model_correlation_ceiling_estimate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def collate(inputs: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in inputs:
        payload = json.loads(path.read_text())
        missing = [field for field in REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"{path} is missing required fields: {', '.join(missing)}.")
        if payload["signal"] != "coverage":
            raise ValueError(f"{path} has signal={payload['signal']!r}, expected 'coverage'.")
        rows.append({field: payload[field] for field in REQUIRED_FIELDS})
    rows.sort(key=lambda row: str(row["species"]))
    species = [str(row["species"]) for row in rows]
    if len(set(species)) != len(species):
        raise ValueError("Each species must have exactly one split-half result.")
    return rows


def markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Johansen ATAC Split-Half Reliability",
        "",
        "Each value is computed from independently depth-balanced cell halves within each pseudobulk group. The full-target reliability uses the Spearman-Brown correction and the ceiling is its square root. Results remain species-specific because their target constructions, depths, and sequence-model endpoints differ.",
        "",
        "| Species | Chromosome | Groups | Split-half double-centered R | Full-target reliability | Estimated model ceiling |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {species} | {chromosome} | {groups} | {split:.4f} | {reliability:.4f} | {ceiling:.4f} |".format(
                species=row["species"],
                chromosome=row["chromosome"],
                groups=row["groups"],
                split=float(row["split_half_double_centered_r"]),
                reliability=float(row["full_target_reliability_estimate"]),
                ceiling=float(row["model_correlation_ceiling_estimate"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = collate(args.input)
    result = {"signal": "coverage", "species": rows}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(markdown(rows))


if __name__ == "__main__":
    main()
