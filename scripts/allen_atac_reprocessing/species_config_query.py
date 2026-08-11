#!/usr/bin/env python3
"""Query fields needed by the Johansen reprocessing launchers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _species_by_name(config_path: Path) -> dict[str, dict]:
    config = json.loads(config_path.read_text())
    return {entry["name"]: entry for entry in config["species"]}


def _chromosomes(entry: dict) -> list[str]:
    configured = entry.get("include_chroms", "")
    if configured:
        return configured.split(",")
    return [f"chr{index}" for index in range(1, 23)] + ["chrX"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    jobs_parser = subparsers.add_parser("chromosome-jobs")
    jobs_parser.add_argument("--species", nargs="+", required=True)

    fasta_parser = subparsers.add_parser("fasta")
    fasta_parser.add_argument("--species", required=True)
    args = parser.parse_args()

    species = _species_by_name(args.species_config)
    if args.command == "chromosome-jobs":
        for name in args.species:
            entry = species[name]
            for chromosome in _chromosomes(entry):
                print(name, chromosome, entry["fasta"], sep="\t")
    else:
        print(species[args.species]["fasta"])


if __name__ == "__main__":
    main()
