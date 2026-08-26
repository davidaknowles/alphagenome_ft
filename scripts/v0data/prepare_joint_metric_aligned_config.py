#!/usr/bin/env python3
"""Prepare the evidence-selected joint metric-aligned objective manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORRELATION_WEIGHTS = {
    "hda": {"hda_rna": 0.1},
    "liu_hdma": {"liu_rna": 1.0},
    "johansen2025": {"allen_rna": 1.0},
    "zemke2023": {"zemke2023_atac": 10.0, "zemke2023_rna": 10.0},
    "zemke2024": {"zemke2024_all_rna": 10.0},
}


def prepare_config(
    source: dict[str, Any],
    *,
    source_path: Path,
    output_dir: Path,
    zemke_weight: float = 10.0,
    sampling_strategy: str | None = None,
) -> dict[str, Any]:
    if not 0 < zemke_weight <= 10:
        raise ValueError(f"zemke_weight must be in (0, 10], got {zemke_weight}.")
    policy_by_dataset = json.loads(json.dumps(CORRELATION_WEIGHTS))
    policy_by_dataset["zemke2023"]["zemke2023_atac"] = zemke_weight
    policy_by_dataset["zemke2023"]["zemke2023_rna"] = zemke_weight
    policy_by_dataset["zemke2024"]["zemke2024_all_rna"] = zemke_weight
    result = json.loads(json.dumps(source))
    if sampling_strategy is not None:
        if sampling_strategy not in {"equal_datasets", "equal_sources"}:
            raise ValueError(
                "sampling_strategy must be 'equal_datasets' or 'equal_sources', "
                f"got {sampling_strategy!r}."
            )
        result["sampling_strategy"] = sampling_strategy
        result["sampling"] = (
            "equal optimizer updates per native source using the largest "
            "single-source batch count as the epoch budget"
            if sampling_strategy == "equal_sources"
            else "equal optimizer updates per dataset using the largest single-source "
            "batch count as the epoch budget; round-robin native-source updates within "
            "each dataset"
        )
    source_root = source_path.resolve().parent
    observed: dict[str, set[str]] = {}
    for dataset in result.get("datasets", ()):
        dataset_name = dataset["name"]
        policy = policy_by_dataset.get(dataset_name)
        if policy is None:
            raise ValueError(f"No metric-alignment policy for dataset {dataset_name!r}.")
        for source_entry in dataset.get("sources", ()):
            source_name = source_entry["name"]
            manifest_path = Path(source_entry["targets_config"]).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = source_root / manifest_path
            manifest = json.loads(manifest_path.read_text())
            heads = {head["id"]: head for head in manifest.get("heads", ())}
            missing = set(policy) - set(heads)
            if missing:
                raise ValueError(f"Missing heads in {manifest_path}: {sorted(missing)}")
            for head_id, weight in policy.items():
                heads[head_id]["double_centered_correlation_loss_weight"] = weight
            observed.setdefault(dataset_name, set()).update(policy)

            destination = (output_dir / dataset_name / source_name / "targets.json").resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(manifest, indent=2) + "\n")
            source_entry["targets_config"] = str(destination)

    expected = {dataset: set(policy) for dataset, policy in policy_by_dataset.items()}
    if observed != expected:
        raise ValueError(f"Incomplete metric-alignment coverage: {observed!r}")
    result["objective_variant"] = {
        "name": "metric_aligned" if zemke_weight == 10 else "metric_tempered",
        "double_centered_correlation_loss_weights": policy_by_dataset,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--zemke-weight", type=float, default=10.0)
    parser.add_argument(
        "--sampling-strategy",
        choices=("equal_datasets", "equal_sources"),
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    result = prepare_config(
        json.loads(source_path.read_text()),
        source_path=source_path,
        output_dir=output_dir,
        zemke_weight=args.zemke_weight,
        sampling_strategy=args.sampling_strategy,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "datasets.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {destination}.")


if __name__ == "__main__":
    main()
