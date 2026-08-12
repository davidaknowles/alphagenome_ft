#!/usr/bin/env python3
"""Audit low-rank cell-group structure in gene-level RNA targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import double_centered_rank_summary


def _parse_dataset(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("Datasets must use LABEL=PATH syntax.")
    return label, Path(path)


def audit(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        cpm = np.asarray(data["cpm"], dtype=np.float64).T
    return {
        "gene_supervision": str(path),
        "raw_cpm": double_centered_rank_summary(cpm),
        "log1p_cpm": double_centered_rank_summary(np.log1p(cpm)),
    }


def _format(value: float) -> str:
    return f"{value:.4f}"


def render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Gene-level RNA target-rank audit",
        "",
        "For a double-centered gene-by-cell-group target matrix, the rank-k ceiling is the correlation with its optimal rank-k singular-value approximation. It measures target representability by a shared cell-group basis, not achievable sequence-model accuracy.",
    ]
    for scale, title in (("raw_cpm", "Raw counts per million"), ("log1p_cpm", "log1p counts per million")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Dataset | Genes | Groups | Effective rank | Rank for R=0.8 | Rank-1 ceiling | Rank-4 ceiling | Rank-8 ceiling | Rank-16 ceiling | Rank-32 ceiling |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, audit_result in result["datasets"].items():
            summary = audit_result[scale]
            ceilings = summary["rank_correlation_ceiling"]
            lines.append(
                f"| {label} | {summary['observations']:,} | {summary['tracks']} | "
                f"{summary['entropy_effective_rank']:.2f} | "
                f"{summary['rank_for_correlation']['0.8']} | "
                + " | ".join(_format(ceilings[str(rank)]) for rank in (1, 4, 8, 16, 32))
                + " |"
            )
    lines.extend(
        [
            "",
            "A low rank requirement would support a factorized cell-group output head. A high requirement would instead favor full channel-specific heads and objective or backbone changes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "definition": "Optimal singular-value approximation after centering genes and cell groups.",
        "datasets": {label: audit(path.expanduser()) for label, path in args.dataset},
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(result))
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
