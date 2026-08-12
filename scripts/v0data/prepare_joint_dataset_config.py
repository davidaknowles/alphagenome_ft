#!/usr/bin/env python
"""Build the final non-ENCODE joint-training dataset configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hda-targets",
        type=Path,
        default=Path("outputs/v0data/hda-joint/targets_geneonly_unstranded.json"),
    )
    parser.add_argument(
        "--liu-targets",
        type=Path,
        default=Path("outputs/v0data/liu-hdma/joint/targets_geneonly_corrw1.json"),
    )
    parser.add_argument(
        "--johansen-species",
        type=Path,
        default=Path("outputs/v0data/johansen-rna-corrected/geneonly-corrw1/species.json"),
    )
    parser.add_argument(
        "--zemke2023-species",
        type=Path,
        default=Path("outputs/v0data/zemke2023-species/species.json"),
    )
    parser.add_argument(
        "--zemke2024-targets",
        type=Path,
        default=Path("outputs/v0data/zemke2024-all/targets.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/v0data/joint-all-nonencode/datasets.json"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_species(path: Path) -> list[dict]:
    payload = json.loads(resolve(path).read_text())
    species = payload.get("species")
    if not species:
        raise ValueError(f"Species configuration contains no entries: {path}")
    return [dict(entry) for entry in species]


def human_source(name: str, fasta: Path, targets: Path) -> dict:
    return {
        "name": name,
        "organism": "HOMO_SAPIENS",
        "fasta": str(fasta),
        "targets_config": str(resolve(targets)),
        "valid_chroms": "chr8",
        "test_chroms": "chr9",
        "exclude_chroms": "chrM,chrY",
        "include_chroms": "",
    }


def main() -> None:
    args = parse_args()
    johansen_sources = load_species(args.johansen_species)
    zemke2023_sources = load_species(args.zemke2023_species)
    human_fasta = resolve(Path(johansen_sources[0]["fasta"]))
    payload = {
        "schema_version": 1,
        "selection_status": "provisional canonical targets pending objective screens",
        "sampling": (
            "equal optimizer updates per dataset using the largest single-source batch "
            "count as the epoch budget; round-robin native-source updates within each dataset"
        ),
        "datasets": [
            {
                "name": "hda",
                "sources": [
                    human_source("human", human_fasta, args.hda_targets)
                ],
            },
            {
                "name": "liu_hdma",
                "sources": [
                    human_source("human", human_fasta, args.liu_targets)
                ],
            },
            {"name": "johansen2025", "sources": johansen_sources},
            {"name": "zemke2023", "sources": zemke2023_sources},
            {
                "name": "zemke2024",
                "sources": [
                    human_source("human", human_fasta, args.zemke2024_targets)
                ],
            },
        ],
    }
    output = args.output.expanduser()
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
