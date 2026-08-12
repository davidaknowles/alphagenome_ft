#!/usr/bin/env python
"""Estimate technical sampling reliability of Mannens RNA pseudobulks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import (
    binomial_count_split,
    counts_per_million,
    double_centered_pearson,
    spearman_brown,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5ad", required=True, type=Path)
    parser.add_argument("--gene-supervision", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    return parser.parse_args()


def _source_gene_ids(data: ad.AnnData) -> tuple[str, ...]:
    if "Accession" not in data.var:
        raise ValueError("RNA AnnData lacks the required var['Accession'] gene identifiers.")
    return tuple(str(value).split(".", 1)[0] for value in data.var["Accession"])


def load_modeled_raw_counts(
    rna_h5ad: Path,
    gene_supervision: Path,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Load raw counts for the exact groups and genes used by the model."""
    with np.load(gene_supervision) as supervision:
        groups = tuple(str(value) for value in supervision["groups"])
        genes = tuple(str(value).split(".", 1)[0] for value in supervision["gene_ids"])

    data = ad.read_h5ad(rna_h5ad, backed="r")
    try:
        source_groups = {str(value): index for index, value in enumerate(data.obs_names)}
        source_genes = {value: index for index, value in enumerate(_source_gene_ids(data))}
        missing_groups = sorted(set(groups) - set(source_groups))
        missing_genes = sorted(set(genes) - set(source_genes))
        if missing_groups or missing_genes:
            raise ValueError(
                f"Missing {len(missing_groups)} modeled groups and {len(missing_genes)} modeled genes."
            )
        row_indices = np.asarray([source_groups[group] for group in groups])
        column_indices = np.asarray([source_genes[gene] for gene in genes])
        # This source matrix is small enough to materialize once, and doing so
        # avoids backend-specific restrictions on two-axis fancy indexing.
        raw = np.asarray(data.layers["raw"][:])
        counts = raw[np.ix_(row_indices, column_indices)]
    finally:
        data.file.close()
    return counts, groups, genes


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def audit_count_sampling_reliability(
    counts: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    raw_correlations: list[float] = []
    log_correlations: list[float] = []
    for repeat in range(repeats):
        first_counts, second_counts = binomial_count_split(counts, seed=seed + repeat)
        first_cpm, first_valid = counts_per_million(first_counts)
        second_cpm, second_valid = counts_per_million(second_counts)
        valid = first_valid & second_valid
        if valid.sum() < 2:
            raise ValueError("Fewer than two groups have molecules in both split halves.")
        raw_correlations.append(double_centered_pearson(first_cpm[valid], second_cpm[valid]))
        log_correlations.append(
            double_centered_pearson(np.log1p(first_cpm[valid]), np.log1p(second_cpm[valid]))
        )

    raw = _summary(raw_correlations)
    log = _summary(log_correlations)
    raw_reliability = spearman_brown(raw["mean"])
    log_reliability = spearman_brown(log["mean"])
    return {
        "repeats": repeats,
        "seed": seed,
        "groups": int(counts.shape[0]),
        "genes": int(counts.shape[1]),
        "total_molecules": int(np.rint(counts).sum()),
        "raw_cpm_split_half_double_centered_r": raw,
        "raw_cpm_full_technical_reliability": raw_reliability,
        "raw_cpm_assumption_based_model_r_ceiling": float(np.sqrt(max(raw_reliability, 0.0))),
        "log1p_cpm_split_half_double_centered_r": log,
        "log1p_cpm_full_technical_reliability": log_reliability,
        "scope": (
            "Repeated binomial splits of aggregate cluster counts estimate technical molecule-sampling "
            "repeatability only. The source lacks donor-resolved counts, so these values do not include "
            "biological donor or specimen variability."
        ),
    }


def _write_markdown(result: dict[str, object], output: Path) -> None:
    raw = result["raw_cpm_split_half_double_centered_r"]
    log = result["log1p_cpm_split_half_double_centered_r"]
    text = (
        "# Mannens RNA technical sampling reliability\n\n"
        "Each published cluster-level raw-count pseudobulk was divided into two equal-probability "
        "molecule samples, independently for every gene. Each half was normalized to counts per "
        "million, CPM, and compared using signed double-centered correlation. Repeated binomial splits "
        "measure technical molecule-sampling repeatability only. The released matrix has no donor-level "
        "counts, so this audit does not measure biological donor or specimen variability.\n\n"
        "| Quantity | Value |\n"
        "|---|---:|\n"
        f"| Modeled cell groups | {result['groups']} |\n"
        f"| Modeled genes | {result['genes']:,} |\n"
        f"| Raw molecules | {result['total_molecules']:,} |\n"
        f"| Repeated splits | {result['repeats']} |\n"
        f"| Raw CPM split-half double-centered R, mean | {raw['mean']:.4f} |\n"
        f"| Raw CPM split-half double-centered R, standard deviation | {raw['standard_deviation']:.4f} |\n"
        f"| Raw CPM full technical reliability, Spearman-Brown | {result['raw_cpm_full_technical_reliability']:.4f} |\n"
        f"| Raw CPM assumption-based model R ceiling | {result['raw_cpm_assumption_based_model_r_ceiling']:.4f} |\n"
        f"| log1p CPM split-half double-centered R, mean | {log['mean']:.4f} |\n"
        f"| log1p CPM split-half double-centered R, standard deviation | {log['standard_deviation']:.4f} |\n"
        f"| log1p CPM full technical reliability, Spearman-Brown | {result['log1p_cpm_full_technical_reliability']:.4f} |\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main() -> None:
    args = _parse_args()
    counts, _, _ = load_modeled_raw_counts(args.rna_h5ad, args.gene_supervision)
    result = audit_count_sampling_reliability(counts, repeats=args.repeats, seed=args.seed)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, args.markdown_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
