#!/usr/bin/env python3
"""Give each native source in selected joint datasets an independent output head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_DATASETS = ("johansen2025", "zemke2023")


def _resolved(path_value: str, source_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = source_root / path
    return path.resolve()


def prepare_config(
    source: dict[str, Any],
    *,
    source_path: Path,
    output_dir: Path,
    dataset_names: Sequence[str] = DEFAULT_DATASETS,
) -> dict[str, Any]:
    """Copy manifests while renaming heads for the requested multispecies datasets."""
    result = json.loads(json.dumps(source))
    selected = set(map(str, dataset_names))
    available = {str(dataset["name"]) for dataset in result.get("datasets", ())}
    missing = selected - available
    if missing:
        raise ValueError(f"Joint configuration lacks datasets: {sorted(missing)}")
    if result.get("sampling_strategy") != "equal_sources":
        raise ValueError("Source-specific heads require equal_sources sampling for fair exposure.")

    source_root = source_path.resolve().parent
    renamed: dict[str, dict[str, list[str]]] = {}
    for dataset in result["datasets"]:
        dataset_name = str(dataset["name"])
        if dataset_name not in selected:
            continue
        sources = dataset.get("sources", ())
        if len(sources) < 2:
            raise ValueError(
                f"Source-specific dataset {dataset_name!r} must have at least two sources."
            )
        dataset["source_specific_heads"] = True
        renamed[dataset_name] = {}
        weights = result.get("objective_variant", {}).get(
            "double_centered_correlation_loss_weights", {}
        ).get(dataset_name)
        original_weights = dict(weights) if isinstance(weights, dict) else {}
        old_head_ids: set[str] = set()
        for source_entry in sources:
            source_name = str(source_entry["name"])
            manifest_path = _resolved(source_entry["targets_config"], source_root)
            manifest = json.loads(manifest_path.read_text())
            old_ids = []
            new_ids = []
            for head in manifest.get("heads", ()):
                old_id = str(head["id"])
                new_id = f"{old_id}_{source_name}"
                head["id"] = new_id
                old_ids.append(old_id)
                new_ids.append(new_id)
                old_head_ids.add(old_id)
            if not new_ids:
                raise ValueError(f"Target manifest contains no heads: {manifest_path}")
            destination = (output_dir / dataset_name / source_name / "targets.json").resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(manifest, indent=2) + "\n")
            source_entry["targets_config"] = str(destination)
            renamed[dataset_name][source_name] = new_ids

            if isinstance(weights, dict):
                for old_id, new_id in zip(old_ids, new_ids, strict=True):
                    if old_id in original_weights:
                        weights[new_id] = original_weights[old_id]
        if isinstance(weights, dict):
            for old_id in old_head_ids:
                weights.pop(old_id, None)

    result["head_routing_variant"] = {
        "name": "source_specific_heads",
        "datasets": sorted(selected),
        "renamed_heads": renamed,
        "shared_backbone": True,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    result = prepare_config(
        json.loads(source_path.read_text()),
        source_path=source_path,
        output_dir=output_dir,
        dataset_names=tuple(filter(None, args.datasets.split(","))),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "datasets.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {destination}.")


if __name__ == "__main__":
    main()
