#!/usr/bin/env python3
"""Audit canonical LoRA and LoRA plus LoCon result coverage by dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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

STRATEGY_SUFFIXES = {
    "lora": "_lora",
    "lora+locon": "_lora_locon",
}
SUPERSEDED_RUNS = frozenset(
    {
        # This checkpoint used summed per-cell-normalized Johansen expression.
        "johansen_joint_lora_locon",
    }
)


def _completed_epochs(path: Path) -> list[int]:
    if path.parent.name in SUPERSEDED_RUNS:
        return []
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
            if isinstance(epoch, int) and record.get("metrics", {}).get("valid"):
                epochs.add(epoch)
    return sorted(epochs)


def audit_coverage(
    checkpoint_root: Path,
    expected_datasets: tuple[str, ...] = EXPECTED_DATASETS,
) -> dict[str, Any]:
    datasets = []
    for dataset in expected_datasets:
        strategy_epochs = {
            strategy: _completed_epochs(
                checkpoint_root / f"{dataset}{suffix}" / "metrics.jsonl"
            )
            for strategy, suffix in STRATEGY_SUFFIXES.items()
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
    return {"datasets": datasets}


def _format_epoch(epoch: int | None) -> str:
    return "" if epoch is None else str(epoch)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Canonical adapter coverage",
        "",
        "A matched result means both strategies completed the same epoch. It does not imply that early stopping completed or that the requested correlation was reached.",
        "",
        "| Dataset | Latest LoRA epoch | Latest LoRA+LoCon epoch | Highest matched epoch | Status |",
        "|---|---:|---:|---:|---|",
    ]
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
