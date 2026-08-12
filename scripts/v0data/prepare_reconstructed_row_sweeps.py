#!/usr/bin/env python3
"""Prepare Liu and Johansen row-correlation sweep manifests in one process."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.target_manifest import make_gene_only_config


WEIGHTS = (("0", 0.0), ("0p1", 0.1), ("1", 1.0), ("10", 10.0))


def main() -> None:
    liu_source_path = Path(
        "outputs/v0data/liu-hdma/joint/targets_geneonly_corrw1.json"
    ).resolve()
    liu_source = json.loads(liu_source_path.read_text())
    johansen_source_path = Path(
        "outputs/v0data/johansen-rna-corrected/geneonly-corrw1/species.json"
    ).resolve()
    johansen_source = json.loads(johansen_source_path.read_text())

    for suffix, weight in WEIGHTS:
        liu_output = liu_source_path.parent / f"targets_geneonly_rowcorrw{suffix}.json"
        liu_targets = make_gene_only_config(
            liu_source,
            head_id="liu_rna",
            correlation_loss_weight=0.0,
            row_correlation_loss_weight=weight,
        )
        liu_output.write_text(json.dumps(liu_targets, indent=2) + "\n")

        output_root = Path(
            f"outputs/v0data/johansen-rna-corrected/geneonly-rowcorrw{suffix}"
        ).resolve()
        species_config = copy.deepcopy(johansen_source)
        for entry in species_config.get("species", ()):
            source_targets = Path(entry["targets_config"]).expanduser()
            if not source_targets.is_absolute():
                source_targets = johansen_source_path.parent / source_targets
            targets = make_gene_only_config(
                json.loads(source_targets.read_text()),
                head_id="allen_rna",
                correlation_loss_weight=0.0,
                row_correlation_loss_weight=weight,
            )
            species_dir = output_root / str(entry["name"])
            species_dir.mkdir(parents=True, exist_ok=True)
            output_targets = species_dir / "targets.json"
            output_targets.write_text(json.dumps(targets, indent=2) + "\n")
            entry["targets_config"] = str(output_targets)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "species.json").write_text(
            json.dumps(species_config, indent=2) + "\n"
        )
        print(f"Prepared row-correlation weight {weight:g}.")


if __name__ == "__main__":
    main()
