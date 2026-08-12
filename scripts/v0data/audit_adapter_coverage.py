#!/usr/bin/env python3
"""Audit canonical LoRA and LoRA plus LoCon result coverage by dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v0data.collate_adapter_comparisons import (
    canonical_run_directory,
    has_complete_correlations,
)
from scripts.v0data.collate_joint_species_evaluations import collate as collate_native_evaluations


EXPECTED_DATASETS = (
    "hda",
    "hda-joint",
    "johansen-human",
    "johansen_joint",
    "liu-hdma",
    "zemke2023-human",
    "zemke2023_macaque",
    "zemke2023_marmoset",
    "zemke2023_mouse",
    "zemke2023_joint",
    "zemke2024-all",
)

PRIMARY_STUDIES = (
    {
        "study": "Mannens HDA",
        "canonical_dataset": "hda-joint",
        "native_species": (),
    },
    {
        "study": "Johansen 2025",
        "canonical_dataset": "johansen_joint",
        "native_species": ("human", "macaque", "marmoset"),
    },
    {
        "study": "Liu HDMA",
        "canonical_dataset": "liu-hdma",
        "native_species": (),
    },
    {
        "study": "Zemke 2023",
        "canonical_dataset": "zemke2023_joint",
        "native_species": ("human", "macaque", "marmoset", "mouse"),
    },
    {
        "study": "Zemke 2024",
        "canonical_dataset": "zemke2024-all",
        "native_species": (),
    },
)

def _completed_epochs(path: Path) -> list[int]:
    if not path.exists():
        return []
    epochs = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            epoch = record.get("epoch")
            if isinstance(epoch, int) and has_complete_correlations(record):
                epochs.add(epoch)
    return sorted(epochs)


def audit_coverage(
    checkpoint_root: Path,
    expected_datasets: tuple[str, ...] = EXPECTED_DATASETS,
    primary_studies: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    datasets = []
    for dataset in expected_datasets:
        strategy_epochs = {
            strategy: _completed_epochs(
                checkpoint_root
                / canonical_run_directory(dataset, strategy)
                / "metrics.jsonl"
            )
            for strategy in ("lora", "lora+locon")
        }
        common_epochs = sorted(
            set(strategy_epochs["lora"]) & set(strategy_epochs["lora+locon"])
        )
        missing = [strategy for strategy, epochs in strategy_epochs.items() if not epochs]
        if missing:
            status = "missing " + ", ".join(missing)
        elif not common_epochs:
            status = "no matched epoch"
        else:
            status = "matched result available"
        datasets.append(
            {
                "dataset": dataset,
                "completed_epochs": strategy_epochs,
                "latest_lora_epoch": max(strategy_epochs["lora"], default=None),
                "latest_lora_locon_epoch": max(
                    strategy_epochs["lora+locon"], default=None
                ),
                "highest_matched_epoch": max(common_epochs, default=None),
                "status": status,
            }
        )
    if primary_studies is None:
        primary_studies = tuple(
            study
            for study in PRIMARY_STUDIES
            if study["canonical_dataset"] in expected_datasets
        )
    coverage_by_dataset = {entry["dataset"]: entry for entry in datasets}
    native_evaluations = collate_native_evaluations(checkpoint_root)["evaluations"]
    native_keys = {
        (evaluation["dataset"], evaluation["species"], evaluation["strategy"])
        for evaluation in native_evaluations
    }
    studies = []
    for study in primary_studies:
        dataset = str(study["canonical_dataset"])
        expected_species = tuple(map(str, study.get("native_species", ())))
        canonical = coverage_by_dataset.get(dataset)
        canonical_status = canonical["status"] if canonical is not None else "not audited"
        missing_native = {
            strategy: [
                species
                for species in expected_species
                if (dataset, species, strategy) not in native_keys
            ]
            for strategy in ("lora", "lora+locon")
        }
        native_complete = not any(missing_native.values())
        if canonical_status != "matched result available":
            status = canonical_status
        elif not native_complete:
            status = "missing native evaluations"
        else:
            status = "comparison coverage complete"
        studies.append(
            {
                "study": str(study["study"]),
                "canonical_dataset": dataset,
                "canonical_status": canonical_status,
                "expected_native_species": list(expected_species),
                "missing_native_evaluations": missing_native,
                "status": status,
            }
        )
    return {"primary_studies": studies, "datasets": datasets}


def _format_epoch(epoch: int | None) -> str:
    return "" if epoch is None else str(epoch)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Canonical adapter coverage",
        "",
        "The primary table covers each requested non-ENCODE study. Cross-species studies additionally require evaluation of both joint adapter checkpoints against every native species. A matched result means both strategies completed the same epoch; it does not imply that early stopping completed or that the requested correlation was reached.",
        "",
        "## Primary studies",
        "",
        "| Study | Canonical comparison | Native species required | Missing native evaluations | Coverage status |",
        "|---|---|---|---|---|",
    ]
    for study in result.get("primary_studies", ()):
        expected_species = ", ".join(study["expected_native_species"]) or "not applicable"
        missing = []
        for strategy in ("lora", "lora+locon"):
            species = study["missing_native_evaluations"][strategy]
            if species:
                missing.append(f"{strategy}: {', '.join(species)}")
        lines.append(
            f"| {study['study']} | `{study['canonical_dataset']}` | {expected_species} | "
            f"{'; '.join(missing)} | {study['status']} |"
        )
    lines.extend(
        [
        "",
        "## All canonical arms",
        "",
        "| Dataset | Latest LoRA epoch | Latest LoRA+LoCon epoch | Highest matched epoch | Status |",
        "|---|---:|---:|---:|---|",
        ]
    )
    for dataset in result["datasets"]:
        lines.append(
            f"| `{dataset['dataset']}` | "
            f"{_format_epoch(dataset['latest_lora_epoch'])} | "
            f"{_format_epoch(dataset['latest_lora_locon_epoch'])} | "
            f"{_format_epoch(dataset['highest_matched_epoch'])} | "
            f"{dataset['status']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/v0data"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    result = audit_coverage(args.checkpoint_root)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(result))
    if not args.json_output and not args.markdown_output:
        print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
