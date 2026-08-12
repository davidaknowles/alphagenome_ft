#!/usr/bin/env python
"""Compare legacy and raw-count-derived Johansen RNA pseudobulks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import read_pseudobulk_expression
from alphagenome_ft.finetune.reliability import double_centered_pearson


def audit_species(species: str, legacy_path: Path, corrected_path: Path) -> dict[str, object]:
    legacy = read_pseudobulk_expression(
        legacy_path,
        normalize_cpm=True,
        gene_id_column="_index",
        group_column=None,
    )
    corrected = read_pseudobulk_expression(
        corrected_path,
        normalize_cpm=False,
        gene_id_column="_index",
        group_column=None,
    )
    if legacy.groups != corrected.groups or legacy.gene_ids != corrected.gene_ids:
        raise ValueError(f"Legacy and corrected labels differ for {species}.")
    per_group = np.asarray(
        [
            np.corrcoef(legacy.cpm[index], corrected.cpm[index])[0, 1]
            for index in range(len(legacy.groups))
        ]
    )
    return {
        "species": species,
        "groups": len(legacy.groups),
        "genes": len(legacy.gene_ids),
        "raw_cpm_double_centered_r": double_centered_pearson(
            legacy.cpm, corrected.cpm
        ),
        "log1p_cpm_double_centered_r": double_centered_pearson(
            np.log1p(legacy.cpm), np.log1p(corrected.cpm)
        ),
        "per_group_pearson_quantiles": {
            str(quantile): float(np.quantile(per_group, quantile))
            for quantile in (0.0, 0.05, 0.5, 0.95, 1.0)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        action="append",
        required=True,
        metavar="NAME=LEGACY=CORRECTED",
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    audits = []
    for value in args.species:
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"Expected NAME=LEGACY=CORRECTED, got {value!r}.")
        audits.append(audit_species(parts[0], Path(parts[1]), Path(parts[2])))

    payload = {
        "definition": (
            "Legacy matrices sum per-cell normalized expression before row CPM; corrected "
            "matrices sum raw UMI counts by group before CPM."
        ),
        "audits": audits,
    }
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Johansen RNA normalization audit",
        "",
        "Legacy pseudobulks summed expression after per-cell normalization. Corrected pseudobulks sum raw unique molecular identifier, UMI, counts within each cell group and then normalize the group total to counts per million, CPM. Agreement is measured after putting both matrices in CPM units.",
        "",
        "| Species | Groups | Genes | Raw CPM double-centered R | log1p CPM double-centered R | Median per-group R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for audit in audits:
        lines.append(
            f"| {audit['species']} | {audit['groups']} | {audit['genes']} | "
            f"{audit['raw_cpm_double_centered_r']:.4f} | "
            f"{audit['log1p_cpm_double_centered_r']:.4f} | "
            f"{audit['per_group_pearson_quantiles']['0.5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The raw-CPM discrepancy changes the cell-group-specific target structure, not only its scale. Johansen RNA training results produced from the legacy matrices are therefore superseded by raw-count-derived targets.",
            "",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
