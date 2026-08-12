#!/usr/bin/env python
"""Estimate Liu RNA pseudobulk reliability from sample-level split halves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import (
    balanced_library_split,
    counts_per_million,
    double_centered_pearson,
    fixed_window_gene_mask,
    spearman_brown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--gene-supervision", type=Path)
    parser.add_argument("--fasta-index", type=Path)
    parser.add_argument("--window-size", type=int, default=131_072)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    return parser.parse_args()


def _row_correlations(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(first * first, axis=1) * np.sum(second * second, axis=1))
    return np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.full((len(first),), np.nan),
        where=denominator > 0,
    )


def chromosome_reliability(
    first_cpm: np.ndarray,
    second_cpm: np.ndarray,
    gene_ids: tuple[str, ...],
    gene_supervision_path: Path,
    *,
    fasta_index_path: Path | None = None,
    window_size: int = 131_072,
) -> dict[str, dict[str, float | int]]:
    """Report split-half reliability for modeled genes on each chromosome."""
    with np.load(gene_supervision_path) as supervision:
        modeled_ids = supervision["gene_ids"].astype(str)
        modeled_chromosomes = supervision["chromosomes"].astype(str)
    if modeled_ids.shape != modeled_chromosomes.shape:
        raise ValueError("Modeled gene IDs and chromosomes must have equal shape.")
    modeled_valid = np.ones(len(modeled_ids), dtype=bool)
    if fasta_index_path is not None:
        with np.load(gene_supervision_path) as supervision:
            modeled_starts = np.asarray(supervision["starts"], dtype=np.int64)
            modeled_ends = np.asarray(supervision["ends"], dtype=np.int64)
        chromosome_sizes = {
            fields[0]: int(fields[1])
            for line in fasta_index_path.read_text().splitlines()
            if len(fields := line.split("\t")) >= 2
        }
        modeled_valid = fixed_window_gene_mask(
            modeled_chromosomes,
            modeled_starts,
            modeled_ends,
            chromosome_sizes,
            window_size=window_size,
            stride=window_size,
        )
    chromosome_by_gene = {
        gene_id.split(".", 1)[0]: chromosome
        for gene_id, chromosome, valid in zip(
            modeled_ids, modeled_chromosomes, modeled_valid, strict=True
        )
        if valid
    }
    normalized_ids = tuple(gene_id.split(".", 1)[0] for gene_id in gene_ids)
    result: dict[str, dict[str, float | int]] = {}
    for chromosome in sorted(set(chromosome_by_gene.values())):
        indices = np.asarray(
            [
                index
                for index, gene_id in enumerate(normalized_ids)
                if chromosome_by_gene.get(gene_id) == chromosome
            ],
            dtype=np.int64,
        )
        if len(indices) < 2:
            continue
        raw_r = double_centered_pearson(first_cpm[:, indices], second_cpm[:, indices])
        log_r = double_centered_pearson(
            np.log1p(first_cpm[:, indices]), np.log1p(second_cpm[:, indices])
        )
        result[chromosome] = {
            "genes": int(len(indices)),
            "raw_cpm_double_centered_r": raw_r,
            "raw_cpm_full_reliability_estimate": spearman_brown(raw_r),
            "log1p_cpm_double_centered_r": log_r,
            "log1p_cpm_full_reliability_estimate": spearman_brown(log_r),
        }
    return result


def main() -> None:
    args = parse_args()
    paths = sorted(args.sample_dir.expanduser().resolve().glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No sample aggregates found under {args.sample_dir}.")

    groups = genes = shape = None
    library_sizes = []
    for path in paths:
        with np.load(path) as sample:
            sample_groups = tuple(sample["groups"].astype(str))
            sample_genes = tuple(sample["gene_ids"].astype(str))
            sample_shape = sample["counts"].shape
            if groups is None:
                groups, genes, shape = sample_groups, sample_genes, sample_shape
            elif (sample_groups, sample_genes, sample_shape) != (groups, genes, shape):
                raise ValueError(f"Sample axes differ in {path}.")
            library_sizes.append(sample["counts"].sum(axis=1, dtype=np.float64))
    assert groups is not None and genes is not None and shape is not None
    library_sizes = np.stack(library_sizes)
    first_assignment = balanced_library_split(library_sizes)

    first_counts = np.zeros(shape, dtype=np.float64)
    second_counts = np.zeros(shape, dtype=np.float64)
    for sample_index, path in enumerate(paths):
        with np.load(path) as sample:
            counts = np.asarray(sample["counts"], dtype=np.float64)
        first_mask = first_assignment[sample_index]
        first_counts[first_mask] += counts[first_mask]
        second_counts[~first_mask] += counts[~first_mask]

    first_cpm, first_valid = counts_per_million(first_counts)
    second_cpm, second_valid = counts_per_million(second_counts)
    valid = first_valid & second_valid
    first_cpm = first_cpm[valid]
    second_cpm = second_cpm[valid]
    raw_r = double_centered_pearson(first_cpm, second_cpm)
    log_r = double_centered_pearson(np.log1p(first_cpm), np.log1p(second_cpm))
    per_group_r = _row_correlations(np.log1p(first_cpm), np.log1p(second_cpm))
    quantiles = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    result = {
        "support": (
            "fixed_window_full_span" if args.fasta_index else "all_supervision_genes"
        ),
        "samples": len(paths),
        "groups": len(groups),
        "genes": len(genes),
        "groups_estimable_in_both_halves": int(valid.sum()),
        "raw_cpm_double_centered_r": raw_r,
        "raw_cpm_full_reliability_estimate": spearman_brown(raw_r),
        "log1p_cpm_double_centered_r": log_r,
        "log1p_cpm_full_reliability_estimate": spearman_brown(log_r),
        "log1p_per_group_pearson_quantiles": {
            str(quantile): float(np.nanquantile(per_group_r, quantile))
            for quantile in quantiles
        },
    }
    if args.gene_supervision:
        result["chromosome_reliability"] = chromosome_reliability(
            first_cpm,
            second_cpm,
            genes,
            args.gene_supervision.expanduser().resolve(),
            fasta_index_path=(
                args.fasta_index.expanduser().resolve() if args.fasta_index else None
            ),
            window_size=args.window_size,
        )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    markdown = (
        "# Liu RNA split-half reliability\n\n"
        f"Support: `{result['support']}`.\n\n"
        "Biological samples were assigned independently for each cell group to two greedily library-depth-balanced halves. Counts were summed and normalized to counts per million, CPM, within each half. The Spearman-Brown correction estimates reliability of the complete pseudobulk from equal-half correlation.\n\n"
        "| Quantity | Value |\n|---|---:|\n"
        f"| Samples | {result['samples']} |\n"
        f"| Cell groups estimable in both halves | {result['groups_estimable_in_both_halves']} / {result['groups']} |\n"
        f"| Genes | {result['genes']} |\n"
        f"| Raw CPM split-half double-centered R | {raw_r:.4f} |\n"
        f"| Raw CPM estimated full-pseudobulk reliability | {result['raw_cpm_full_reliability_estimate']:.4f} |\n"
        f"| log1p CPM split-half double-centered R | {log_r:.4f} |\n"
        f"| log1p CPM estimated full-pseudobulk reliability | {result['log1p_cpm_full_reliability_estimate']:.4f} |\n"
    )
    chromosome_rows = result.get("chromosome_reliability")
    if chromosome_rows:
        markdown += (
            "\n## Chromosome strata\n\n"
            "CPM normalization remains genome-wide before genes are stratified. This table therefore measures target reliability on each chromosome in the same expression units used for evaluation.\n\n"
            "| Chromosome | Genes | Raw CPM split-half R | Raw CPM full reliability | log1p CPM split-half R | log1p CPM full reliability |\n"
            "|---|---:|---:|---:|---:|---:|\n"
        )
        for chromosome, row in chromosome_rows.items():
            markdown += (
                f"| {chromosome} | {row['genes']} | "
                f"{row['raw_cpm_double_centered_r']:.4f} | "
                f"{row['raw_cpm_full_reliability_estimate']:.4f} | "
                f"{row['log1p_cpm_double_centered_r']:.4f} | "
                f"{row['log1p_cpm_full_reliability_estimate']:.4f} |\n"
            )
    args.markdown_output.write_text(markdown)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
