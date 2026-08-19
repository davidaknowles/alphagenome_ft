#!/usr/bin/env python3
"""Copy a joint dataset configuration and update every RNA head objective."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def prepare_config(
    source: dict[str, Any],
    *,
    source_path: Path,
    output_dir: Path,
    loss_weight: float | None,
    correlation_loss_weight: float | None,
) -> dict[str, Any]:
    if loss_weight is None and correlation_loss_weight is None:
        raise ValueError("At least one RNA objective weight must be provided.")
    if loss_weight is not None and (not math.isfinite(loss_weight) or loss_weight <= 0):
        raise ValueError("RNA loss weight must be finite and positive.")
    if correlation_loss_weight is not None and (
        not math.isfinite(correlation_loss_weight) or correlation_loss_weight < 0
    ):
        raise ValueError("RNA correlation loss weight must be finite and non-negative.")

    result = json.loads(json.dumps(source))
    source_root = source_path.resolve().parent
    updated_heads: set[str] = set()
    for dataset in result.get("datasets", ()):
        dataset_name = dataset["name"]
        for source_entry in dataset.get("sources", ()):
            source_name = source_entry["name"]
            manifest_path = Path(source_entry["targets_config"]).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = source_root / manifest_path
            manifest = json.loads(manifest_path.read_text())
            matches = [head for head in manifest.get("heads", ()) if head["id"].endswith("_rna")]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one RNA head in {manifest_path}, found {len(matches)}."
                )
            head = matches[0]
            if loss_weight is not None:
                head["loss_weight"] = loss_weight
            if correlation_loss_weight is not None:
                head["double_centered_correlation_loss_weight"] = correlation_loss_weight
            updated_heads.add(head["id"])

            destination = (output_dir / dataset_name / source_name / "targets.json").resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(manifest, indent=2) + "\n")
            source_entry["targets_config"] = str(destination)

    if not updated_heads:
        raise ValueError("No RNA heads were updated.")
    result["objective_variant"] = {
        "rna_loss_weight": loss_weight,
        "rna_double_centered_correlation_loss_weight": correlation_loss_weight,
        "updated_heads": sorted(updated_heads),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--loss-weight", type=float)
    parser.add_argument("--correlation-loss-weight", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    result = prepare_config(
        json.loads(source_path.read_text()),
        source_path=source_path,
        output_dir=output_dir,
        loss_weight=args.loss_weight,
        correlation_loss_weight=args.correlation_loss_weight,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "datasets.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {destination}.")


if __name__ == "__main__":
    main()
