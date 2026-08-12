#!/usr/bin/env python3
"""Create gene-only RNA target manifests for a multi-species configuration."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.target_manifest import make_gene_only_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--correlation-loss-weight", type=float, default=0.0)
    args = parser.parse_args()

    source_path = args.input.expanduser().resolve()
    source = json.loads(source_path.read_text())
    result = copy.deepcopy(source)
    for entry in result.get("species", ()):
        target_path = Path(entry["targets_config"]).expanduser()
        if not target_path.is_absolute():
            target_path = source_path.parent / target_path
        targets = make_gene_only_config(
            json.loads(target_path.read_text()),
            head_id=args.head,
            correlation_loss_weight=args.correlation_loss_weight,
        )
        species_dir = args.output_dir / str(entry["name"])
        species_dir.mkdir(parents=True, exist_ok=True)
        output_path = species_dir / "targets.json"
        output_path.write_text(json.dumps(targets, indent=2) + "\n")
        entry["targets_config"] = str(output_path.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_config = args.output_dir / "species.json"
    output_config.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote gene-only species configuration to {output_config}.")


if __name__ == "__main__":
    main()
