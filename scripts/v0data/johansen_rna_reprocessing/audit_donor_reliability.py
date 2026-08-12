#!/usr/bin/env python
"""Estimate Johansen RNA reliability from donor-level split halves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from anndata.io import read_elem

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.reliability import (
    double_centered_pearson,
    spearman_brown,
    split_half_pseudobulks,
)
from alphagenome_ft.finetune.reprocessing import aggregate_sparse_count_chunks_by_group
from scripts.v0data.johansen_rna_reprocessing.aggregate_corrected_pseudobulk import (
    _iter_h5ad_csr_rows,
    _read_h5ad_column,
)


def _decode(values) -> tuple[str, ...]:
    return tuple(value.decode() if isinstance(value, bytes) else str(value) for value in values)


def _macaque_gene_mapping(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {
            row["macaque_ensembl"].split(".", 1)[0]: row["macaque_gene"]
            for row in csv.DictReader(handle)
            if row["macaque_ensembl"] and row["macaque_gene"]
        }


def audit_reliability(
    *,
    species: str,
    expression_path: Path,
    template_path: Path,
    orthologs_path: Path,
    chunk_size: int,
) -> dict[str, object]:
    with np.load(template_path, allow_pickle=False) as template:
        target_groups = tuple(template["groups"].astype(str))
        target_gene_ids = tuple(template["gene_ids"].astype(str))

    with h5py.File(expression_path, "r") as source:
        cell_groups = _decode(_read_h5ad_column(source["obs"], "Group"))
        donors = _decode(_read_h5ad_column(source["obs"], "donor_id"))
        var = read_elem(source["raw/var"])
        source_gene_ids = tuple(
            var["ensembl_id"].astype(str)
            if species == "human"
            else var.index.astype(str)
        )
        combined_labels = tuple(
            f"{group}\x1f{donor}" for group, donor in zip(cell_groups, donors, strict=True)
        )
        combined_groups, aggregated, _ = aggregate_sparse_count_chunks_by_group(
            _iter_h5ad_csr_rows(source["raw/X"], chunk_size), combined_labels
        )

    source_gene_index = {gene_id: index for index, gene_id in enumerate(source_gene_ids)}
    if len(source_gene_index) != len(source_gene_ids):
        raise ValueError("Source gene identifiers must be unique.")
    mapped_gene_ids = target_gene_ids
    if species == "macaque":
        mapping = _macaque_gene_mapping(orthologs_path)
        mapped_gene_ids = tuple(mapping.get(gene_id, gene_id) for gene_id in target_gene_ids)
    missing_genes = sorted(set(mapped_gene_ids) - set(source_gene_index))
    if missing_genes:
        raise ValueError(f"Source expression lacks modeled genes: {missing_genes[:10]}")
    gene_columns = np.asarray([source_gene_index[gene] for gene in mapped_gene_ids])

    donor_names = tuple(sorted(set(donors)))
    donor_index = {donor: index for index, donor in enumerate(donor_names)}
    group_index = {group: index for index, group in enumerate(target_groups)}
    counts = np.zeros(
        (len(donor_names), len(target_groups), len(target_gene_ids)), dtype=np.float64
    )
    for row, combined in enumerate(combined_groups):
        group, donor = combined.split("\x1f", 1)
        target_group = group_index.get(group)
        if target_group is not None:
            counts[donor_index[donor], target_group] = aggregated[row, gene_columns]

    first, second, valid = split_half_pseudobulks(counts)
    first = first[valid]
    second = second[valid]
    raw_r = double_centered_pearson(first, second)
    log_r = double_centered_pearson(np.log1p(first), np.log1p(second))
    donor_support = np.sum(counts.sum(axis=2) > 0, axis=0)
    return {
        "species": species,
        "donors": len(donor_names),
        "groups": len(target_groups),
        "genes": len(target_gene_ids),
        "groups_estimable_in_both_halves": int(valid.sum()),
        "donor_support_quantiles": {
            str(q): float(np.quantile(donor_support, q)) for q in (0, 0.25, 0.5, 0.75, 1)
        },
        "raw_cpm_double_centered_r": raw_r,
        "raw_cpm_full_reliability_estimate": spearman_brown(raw_r),
        "raw_cpm_model_correlation_ceiling_estimate": float(
            np.sqrt(max(spearman_brown(raw_r), 0.0))
        ),
        "log1p_cpm_double_centered_r": log_r,
        "log1p_cpm_full_reliability_estimate": spearman_brown(log_r),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True, choices=("human", "macaque", "marmoset"))
    parser.add_argument("--expression", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--orthologs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args()
    result = audit_reliability(
        species=args.species,
        expression_path=args.expression,
        template_path=args.template,
        orthologs_path=args.orthologs,
        chunk_size=args.chunk_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
