#!/usr/bin/env python
"""Point a Johansen species config at corrected gene-only RNA supervision."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.target_manifest import make_gene_only_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--supervision-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--correlation-loss-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.correlation_loss_weight < 0:
        raise ValueError("Correlation loss weight must be non-negative.")

    source_path = args.input.expanduser().resolve()
    source = json.loads(source_path.read_text())
    output_dir = args.output_dir.expanduser().resolve()
    output_entries = []
    for entry in source["species"]:
        species = str(entry["name"])
        targets_path = Path(entry["targets_config"]).expanduser().resolve()
        targets = json.loads(targets_path.read_text())
        rna_heads = [head for head in targets["heads"] if head.get("kind") == "rna_seq"]
        if len(rna_heads) != 1 or "gene_supervision" not in rna_heads[0]:
            raise ValueError(f"Expected one gene-supervised RNA head for {species}.")
        supervision = (args.supervision_root / species / "gene_expression_supervision.npz").resolve()
        if not supervision.exists():
            raise FileNotFoundError(supervision)
        targets = make_gene_only_config(
            targets,
            head_id=str(rna_heads[0]["id"]),
            correlation_loss_weight=args.correlation_loss_weight,
            gene_supervision_path=str(supervision),
        )

        species_dir = output_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)
        corrected_targets = species_dir / "targets.json"
        corrected_targets.write_text(json.dumps(targets, indent=2) + "\n")
        corrected_entry = copy.deepcopy(entry)
        corrected_entry["targets_config"] = str(corrected_targets.resolve())
        output_entries.append(corrected_entry)

    result = copy.deepcopy(source)
    result["species"] = output_entries
    result["rna_source"] = "raw UMI counts summed by cell group, then counts per million"
    result["rna_coverage_loss_weight"] = 0.0
    result["rna_double_centered_correlation_loss_weight"] = args.correlation_loss_weight
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "species.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote corrected Johansen species configuration to {output}.")


if __name__ == "__main__":
    main()
