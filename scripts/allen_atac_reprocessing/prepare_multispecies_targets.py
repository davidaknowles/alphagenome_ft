#!/usr/bin/env python
"""Replace Johansen released ATAC heads with fragment-derived targets."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-species-config", required=True, type=Path)
    parser.add_argument("--atac-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def replace_atac_head(source: dict, replacement: dict, *, species: str) -> dict:
    if len(source.get("heads", ())) != 2:
        raise ValueError(f"Expected two source heads for {species}.")
    if len(replacement.get("heads", ())) != 1:
        raise ValueError(f"Expected one replacement ATAC head for {species}.")
    source_atac = source["heads"][0]
    replacement_atac = replacement["heads"][0]
    replacement_by_label = {target["label"]: target for target in replacement_atac["targets"]}
    source_labels = [target["label"] for target in source_atac["targets"]]
    missing = sorted(set(source_labels) - set(replacement_by_label))
    if missing:
        raise ValueError(f"Fragment-derived {species} ATAC lacks groups {missing}.")

    payload = copy.deepcopy(source)
    payload["heads"][0] = copy.deepcopy(replacement_atac)
    payload["heads"][0]["id"] = source_atac["id"]
    payload["heads"][0]["targets"] = [replacement_by_label[label] for label in source_labels]
    return payload


def main() -> None:
    args = parse_args()
    source_config_path = args.source_species_config.expanduser().resolve()
    source_config = json.loads(source_config_path.read_text())
    source_root = source_config_path.parent
    atac_root = args.atac_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    output_entries = []
    summaries = []
    for entry in source_config["species"]:
        species = str(entry["name"])
        source_targets_path = Path(entry["targets_config"]).expanduser()
        if not source_targets_path.is_absolute():
            source_targets_path = source_root / source_targets_path
        replacement_path = atac_root / species / "bigwigs" / "targets.json"
        source_targets = json.loads(source_targets_path.read_text())
        replacement_targets = json.loads(replacement_path.read_text())
        targets = replace_atac_head(source_targets, replacement_targets, species=species)

        species_dir = output_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)
        targets_path = species_dir / "targets.json"
        targets_path.write_text(json.dumps(targets, indent=2) + "\n")
        output_entry = copy.deepcopy(entry)
        output_entry["targets_config"] = str(targets_path.resolve())
        output_entries.append(output_entry)
        summaries.append(
            {
                "species": species,
                "atac_tracks": len(targets["heads"][0]["targets"]),
                "rna_tracks": len(targets["heads"][1]["targets"]),
                "atac_source": "all-fragment coverage SPMR",
            }
        )

    output_config = copy.deepcopy(source_config)
    output_config["species"] = output_entries
    output_config["atac_source"] = "all-fragment coverage signal per million reads"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "species.json").write_text(json.dumps(output_config, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
