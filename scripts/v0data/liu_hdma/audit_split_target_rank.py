#!/usr/bin/env python3
"""Compare RNA target complexity across held-out Liu chromosomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import double_centered_rank_summary


def audit_chromosomes(path: Path, chromosomes: tuple[str, ...]) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as supervision:
        gene_chromosomes = supervision["chromosomes"].astype(str)
        cpm = np.asarray(supervision["cpm"], dtype=np.float64).T
    if len(gene_chromosomes) != len(cpm):
        raise ValueError("Gene chromosome and CPM axes do not match.")
    results = {}
    for chromosome in chromosomes:
        values = cpm[gene_chromosomes == chromosome]
        if min(values.shape) < 2:
            raise ValueError(f"Chromosome {chromosome} has insufficient target data.")
        results[chromosome] = {
            "raw_cpm": double_centered_rank_summary(values),
            "log1p_cpm": double_centered_rank_summary(np.log1p(values)),
        }
    return {
        "definition": "Optimal singular-value approximation after centering genes and cell groups within each chromosome.",
        "gene_supervision": str(path),
        "chromosomes": results,
    }


def render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Liu held-out chromosome RNA target rank",
        "",
        "For each target matrix $Y\\in\\mathbb{R}^{G\\times C}$, $G$ is genes on one chromosome and $C$ is modeled cell groups. Both axes are centered before singular-value decomposition. Rank ceilings describe target structure, not achievable sequence-model accuracy.",
    ]
    for scale, title in (("raw_cpm", "Raw counts per million"), ("log1p_cpm", "log1p counts per million")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Chromosome | Genes | Groups | Effective rank | Rank for R=0.8 | Rank-2 ceiling | Rank-8 ceiling | Rank-16 ceiling |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for chromosome, chromosome_result in result["chromosomes"].items():
            summary = chromosome_result[scale]
            ceilings = summary["rank_correlation_ceiling"]
            lines.append(
                f"| {chromosome} | {summary['observations']:,} | {summary['tracks']} | "
                f"{summary['entropy_effective_rank']:.2f} | {summary['rank_for_correlation']['0.8']} | "
                f"{ceilings['2']:.4f} | {ceilings['8']:.4f} | {ceilings['16']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-supervision", required=True, type=Path)
    parser.add_argument("--chromosomes", default="chr8,chr9")
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    chromosomes = tuple(value for value in args.chromosomes.split(",") if value)
    result = audit_chromosomes(args.gene_supervision, chromosomes)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(result))
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
