#!/usr/bin/env python
"""Replace Johansen released ATAC heads with fragment-derived targets."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-species-config", required=True, type=Path)
    parser.add_argument("--atac-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--fragment-shards",
        action="append",
        default=[],
        metavar="SPECIES=DIR",
        help="Species and fragment-shard directory used for depth filtering.",
    )
    parser.add_argument("--minimum-fragments", type=int, default=0)
    return parser.parse_args()


def parse_species_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        species, separator, path = value.partition("=")
        if not separator or not species or not path:
            raise ValueError(f"Expected SPECIES=DIR, got {value!r}.")
        if species in result:
            raise ValueError(f"Duplicate fragment-shard species {species!r}.")
        result[species] = Path(path).expanduser().resolve()
    return result


def read_fragment_depths(shard_dir: Path) -> dict[str, int]:
    paths = sorted(shard_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No fragment shards found in {shard_dir}.")
    with np.load(paths[0], allow_pickle=False) as shard:
        groups = shard["groups"].astype(str).tolist()
        totals = shard["total_fragments"].astype(np.int64)
    if len(groups) != len(totals) or len(set(groups)) != len(groups):
        raise ValueError(f"Invalid group/depth arrays in {paths[0]}.")
    return dict(zip(groups, totals.tolist(), strict=True))


def filter_gene_supervision(
    source_path: Path,
    output_path: Path,
    retained_groups: list[str],
) -> None:
    with np.load(source_path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    groups = payload["groups"].astype(str).tolist()
    index_by_group = {group: index for index, group in enumerate(groups)}
    missing = sorted(set(retained_groups) - set(index_by_group))
    if missing:
        raise ValueError(f"Gene supervision lacks retained groups {missing}.")
    indices = np.asarray([index_by_group[group] for group in retained_groups])
    cpm = payload["cpm"]
    if cpm.shape[0] != len(groups):
        raise ValueError(f"Gene CPM rows do not match groups in {source_path}.")
    payload["groups"] = np.asarray(retained_groups, dtype=payload["groups"].dtype)
    payload["cpm"] = cpm[indices]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def replace_atac_head(
    source: dict,
    replacement: dict,
    *,
    species: str,
    retained_groups: list[str] | None = None,
) -> dict:
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

    retained_groups = source_labels if retained_groups is None else retained_groups
    unknown = sorted(set(retained_groups) - set(source_labels))
    if unknown:
        raise ValueError(f"Retained {species} groups are absent from the source: {unknown}.")
    payload = copy.deepcopy(source)
    payload["heads"][0] = copy.deepcopy(replacement_atac)
    payload["heads"][0]["id"] = source_atac["id"]
    payload["heads"][0]["targets"] = [
        replacement_by_label[label] for label in retained_groups
    ]

    source_rna = payload["heads"][1]
    rna_by_label = {target["label"]: target for target in source_rna["targets"]}
    retained_rna_labels = [
        f"{group} ({strand})" for group in retained_groups for strand in ("+", "-")
    ]
    missing_rna = sorted(set(retained_rna_labels) - set(rna_by_label))
    if missing_rna:
        raise ValueError(f"RNA targets for {species} lack labels {missing_rna}.")
    source_rna["targets"] = [rna_by_label[label] for label in retained_rna_labels]
    return payload


def update_shared_group_metadata(config: dict, retained_group_count: int) -> None:
    config["shared_groups"] = retained_group_count
    for summary in config.get("summary", ()):  # Older source configs may omit summaries.
        summary["groups"] = retained_group_count


def main() -> None:
    args = parse_args()
    source_config_path = args.source_species_config.expanduser().resolve()
    source_config = json.loads(source_config_path.read_text())
    source_root = source_config_path.parent
    atac_root = args.atac_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    shard_dirs = parse_species_paths(args.fragment_shards)
    if args.minimum_fragments < 0:
        raise ValueError("--minimum-fragments must be non-negative.")

    source_records = []
    common_groups = None
    depth_by_species = {}
    for entry in source_config["species"]:
        species = str(entry["name"])
        source_targets_path = Path(entry["targets_config"]).expanduser()
        if not source_targets_path.is_absolute():
            source_targets_path = source_root / source_targets_path
        source_targets = json.loads(source_targets_path.read_text())
        groups = [target["label"] for target in source_targets["heads"][0]["targets"]]
        if common_groups is None:
            common_groups = groups
        elif groups != common_groups:
            raise ValueError(f"ATAC group order differs for {species}.")
        if args.minimum_fragments:
            if species not in shard_dirs:
                raise ValueError(f"Missing --fragment-shards entry for {species}.")
            depth_by_species[species] = read_fragment_depths(shard_dirs[species])
        source_records.append((entry, species, source_targets))

    assert common_groups is not None
    retained_groups = [
        group
        for group in common_groups
        if all(
            depth_by_species[species].get(group, 0) >= args.minimum_fragments
            for _, species, _ in source_records
        )
    ]
    if not retained_groups:
        raise ValueError("Fragment-depth filtering removed every shared group.")
    excluded_groups = [group for group in common_groups if group not in retained_groups]

    output_entries = []
    summaries = []
    for entry, species, source_targets in source_records:
        replacement_path = atac_root / species / "bigwigs" / "targets.json"
        replacement_targets = json.loads(replacement_path.read_text())
        targets = replace_atac_head(
            source_targets,
            replacement_targets,
            species=species,
            retained_groups=retained_groups,
        )

        species_dir = output_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)
        source_gene_path = Path(targets["heads"][1]["gene_supervision"]["path"])
        filtered_gene_path = species_dir / "gene_expression_supervision.npz"
        filter_gene_supervision(source_gene_path, filtered_gene_path, retained_groups)
        targets["heads"][1]["gene_supervision"]["path"] = str(
            filtered_gene_path.resolve()
        )
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
                "minimum_fragments": args.minimum_fragments,
                "excluded_groups": {
                    group: depth_by_species.get(species, {}).get(group)
                    for group in excluded_groups
                },
            }
        )

    output_config = copy.deepcopy(source_config)
    output_config["species"] = output_entries
    update_shared_group_metadata(output_config, len(retained_groups))
    output_config["atac_source"] = "all-fragment coverage signal per million reads"
    output_config["minimum_fragments_per_group"] = args.minimum_fragments
    output_config["excluded_groups"] = excluded_groups
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "species.json").write_text(json.dumps(output_config, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
