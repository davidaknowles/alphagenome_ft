#!/usr/bin/env python3
"""Audit held-out RNA target projection onto training-chromosome cell-group factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import (
    double_centered_transfer_rank_summary,
    fixed_window_gene_mask,
)


def audit_transfer_basis(
    gene_supervision: Path,
    fasta_index: Path,
    held_out_chromosomes: tuple[str, ...],
    *,
    window_size: int,
) -> dict[str, object]:
    with np.load(gene_supervision, allow_pickle=False) as data:
        chromosomes = data["chromosomes"].astype(str)
        starts = np.asarray(data["starts"], dtype=np.int64)
        ends = np.asarray(data["ends"], dtype=np.int64)
        values = np.asarray(data["cpm"], dtype=np.float64).T
    chromosome_sizes = {
        fields[0]: int(fields[1])
        for line in fasta_index.read_text().splitlines()
        if len(fields := line.split("\t")) >= 2
    }
    support = fixed_window_gene_mask(
        chromosomes,
        starts,
        ends,
        chromosome_sizes,
        window_size=window_size,
        stride=window_size,
    )
    training_mask = support & ~np.isin(chromosomes, held_out_chromosomes)
    training_values = values[training_mask]
    results = {}
    for chromosome in held_out_chromosomes:
        evaluation_values = values[support & (chromosomes == chromosome)]
        results[chromosome] = {
            "raw_cpm": double_centered_transfer_rank_summary(
                training_values, evaluation_values
            ),
            "log1p_cpm": double_centered_transfer_rank_summary(
                np.log1p(training_values), np.log1p(evaluation_values)
            ),
        }
    return {
        "definition": "Held-out target projection onto right singular vectors learned from double-centered training-chromosome targets.",
        "support": "fixed_window_full_span",
        "gene_supervision": str(gene_supervision),
        "held_out_chromosomes": list(held_out_chromosomes),
        "training_genes": int(training_mask.sum()),
        "chromosomes": results,
    }


def render_markdown(result: dict[str, object], dataset_label: str) -> str:
    lines = [
        f"# {dataset_label} RNA training-basis transfer audit",
        "",
        "The ceiling projects each held-out chromosome target onto cell-group factors learned only from training chromosomes. It measures cross-chromosome target structure and does not use sequence or model predictions.",
        "",
        f"Training genes: {result['training_genes']:,}. Support: `{result['support']}`.",
    ]
    for scale, title in (("raw_cpm", "Raw counts per million"), ("log1p_cpm", "log1p counts per million")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Chromosome | Genes | Groups | Rank-1 | Rank-2 | Rank-4 | Rank-8 | Rank-16 | Rank-32 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for chromosome, entry in result["chromosomes"].items():
            summary = entry[scale]
            ceilings = summary["rank_correlation_ceiling"]
            lines.append(
                f"| {chromosome} | {summary['evaluation_observations']:,} | {summary['tracks']} | "
                + " | ".join(f"{ceilings[str(rank)]:.4f}" for rank in (1, 2, 4, 8, 16, 32))
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-supervision", required=True, type=Path)
    parser.add_argument("--fasta-index", required=True, type=Path)
    parser.add_argument("--held-out-chromosomes", default="chr8,chr9")
    parser.add_argument("--window-size", type=int, default=131_072)
    parser.add_argument("--dataset-label", default="Dataset")
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    held_out = tuple(value for value in args.held_out_chromosomes.split(",") if value)
    result = audit_transfer_basis(
        args.gene_supervision,
        args.fasta_index,
        held_out,
        window_size=args.window_size,
    )
    markdown = render_markdown(result, args.dataset_label)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
