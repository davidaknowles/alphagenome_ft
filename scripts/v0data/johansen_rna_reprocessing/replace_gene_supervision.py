#!/usr/bin/env python
"""Replace a Johansen supervision artifact with raw-count-derived CPM."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import read_pseudobulk_expression
from alphagenome_ft.finetune.reprocessing import align_cpm_to_gene_supervision


def _gene_mapping(path: Path, species: str) -> dict[str, str]:
    if species != "macaque":
        return {}
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["macaque_ensembl"].split(".", 1)[0]: row["macaque_gene"]
            for row in rows
            if row["macaque_ensembl"] and row["macaque_gene"]
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True, choices=("human", "macaque", "marmoset"))
    parser.add_argument("--expression", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--orthologs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    expression = read_pseudobulk_expression(
        args.expression,
        normalize_cpm=False,
        gene_id_column="ensembl_id" if args.species == "human" else "_index",
        group_column=None,
    )
    with np.load(args.template, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["cpm"] = align_cpm_to_gene_supervision(
        template_groups=payload["groups"].astype(str),
        template_gene_ids=payload["gene_ids"].astype(str),
        source_groups=expression.groups,
        source_gene_ids=expression.gene_ids,
        source_cpm=expression.cpm,
        source_gene_by_template=_gene_mapping(args.orthologs, args.species),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(
        f"Wrote corrected {args.species} supervision with shape "
        f"{payload['cpm'].shape} to {args.output}."
    )


if __name__ == "__main__":
    main()
