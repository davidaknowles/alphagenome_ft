#!/usr/bin/env python
"""Collate donor split-half RNA reliability across species."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_SPECIES = ("human", "macaque", "marmoset")


def collate(
    paths: list[Path],
    expected_species: tuple[str, ...] = EXPECTED_SPECIES,
) -> dict[str, Any]:
    if not expected_species or len(set(expected_species)) != len(expected_species):
        raise ValueError("Expected species must be nonempty and unique.")
    audits = [json.loads(path.read_text()) for path in paths]
    by_species = {audit["species"]: audit for audit in audits}
    if len(by_species) != len(audits):
        raise ValueError("Reliability inputs contain duplicate species.")
    missing = set(expected_species) - set(by_species)
    unexpected = set(by_species) - set(expected_species)
    if missing or unexpected:
        raise ValueError(
            f"Expected {expected_species}; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )
    return {
        "definition": (
            "Donor-balanced split-half signed double-centered correlation over raw-count "
            "pseudobulk CPM restricted to retained groups and modeled genes."
        ),
        "audits": [by_species[species] for species in expected_species],
    }


def render_markdown(
    result: dict[str, Any],
    *,
    title: str = "Johansen RNA donor reliability",
    qualification: str | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        "Raw unique molecular identifier counts are aggregated separately by donor and retained cell group. Donors are assigned within each group to library-depth-balanced halves, and each half is normalized to counts per million, CPM. Full reliability uses the Spearman-Brown correction. The model correlation ceiling is the square root of reliability under a classical independent measurement-error assumption; it is not an observed model result.",
        "",
    ]
    if qualification:
        lines.extend((qualification, ""))
    lines.extend(
        (
            "| Species | Donors | Estimable groups | Genes | Split-half raw CPM R | Full reliability | Estimated model R ceiling | Split-half log1p CPM R |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for audit in result["audits"]:
        lines.append(
            f"| {audit['species']} | {audit['donors']} | "
            f"{audit['groups_estimable_in_both_halves']} / {audit['groups']} | "
            f"{audit['genes']} | {audit['raw_cpm_double_centered_r']:.4f} | "
            f"{audit['raw_cpm_full_reliability_estimate']:.4f} | "
            f"{audit['raw_cpm_model_correlation_ceiling_estimate']:.4f} | "
            f"{audit['log1p_cpm_double_centered_r']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    parser.add_argument(
        "--expected-species",
        default=",".join(EXPECTED_SPECIES),
        help="Comma-separated species in output order.",
    )
    parser.add_argument("--title", default="Johansen RNA donor reliability")
    parser.add_argument("--qualification")
    args = parser.parse_args()
    expected_species = tuple(
        species.strip() for species in args.expected_species.split(",") if species.strip()
    )
    result = collate(args.inputs, expected_species)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(
        render_markdown(
            result,
            title=args.title,
            qualification=args.qualification,
        )
    )


if __name__ == "__main__":
    main()
