#!/usr/bin/env python3
"""Collate best-epoch LoRA and LoRA plus LoCon comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

STRATEGY_SUFFIXES = (
    ("_lora_locon", "lora+locon"),
    ("_lora", "lora"),
)


def _run_identity(name: str) -> tuple[str, str] | None:
    if "smoke" in name:
        return None
    for suffix, strategy in STRATEGY_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], strategy
    return None


def _read_epochs(path: Path) -> list[dict[str, Any]]:
    epochs = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                epochs.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return epochs


def _mean_valid_r(record: dict[str, Any]) -> float:
    heads = record.get("metrics", {}).get("valid", {})
    values = [
        metrics.get("differential_pearson_r")
        for metrics in heads.values()
        if isinstance(metrics.get("differential_pearson_r"), (int, float))
        and math.isfinite(metrics["differential_pearson_r"])
    ]
    return sum(values) / len(values) if values else -math.inf


def collate(checkpoint_root: Path) -> dict[str, Any]:
    runs = []
    for metrics_path in sorted(checkpoint_root.glob("*/metrics.jsonl")):
        identity = _run_identity(metrics_path.parent.name)
        if identity is None:
            continue
        epochs = _read_epochs(metrics_path)
        if not epochs:
            continue
        best = max(epochs, key=_mean_valid_r)
        score = _mean_valid_r(best)
        if not math.isfinite(score):
            continue
        dataset, strategy = identity
        heads = []
        split_metrics = best["metrics"]
        for head in sorted(
            set(split_metrics.get("valid", {})) | set(split_metrics.get("test", {}))
        ):
            heads.append(
                {
                    "head": head,
                    "valid_r": split_metrics.get("valid", {})
                    .get(head, {})
                    .get("differential_pearson_r"),
                    "test_r": split_metrics.get("test", {})
                    .get(head, {})
                    .get("differential_pearson_r"),
                }
            )
        runs.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "selected_epoch": best.get("epoch"),
                "global_step": best.get("global_step"),
                "selection_mean_valid_r": score,
                "heads": heads,
                "metrics_path": str(metrics_path),
            }
        )
    return {"selection_metric": "mean valid differential_pearson_r", "runs": runs}


def _format_value(value: Any) -> str:
    return "" if not isinstance(value, (int, float)) or not math.isfinite(value) else f"{value:.4f}"


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Adapter comparison results",
        "",
        "Each run is selected by the mean validation signed double-centered Pearson correlation across its heads. Per-head test values come from that same epoch. Runs can have different maximum epochs while training is active, so strategy conclusions require a matched-epoch comparison.",
        "",
        "| Dataset | Strategy | Epoch | Head | Validation R | Test R |",
        "|---|---|---:|---|---:|---:|",
    ]
    for run in results["runs"]:
        for head in run["heads"]:
            lines.append(
                f"| `{run['dataset']}` | `{run['strategy']}` | {run['selected_epoch']} | `{head['head']}` | "
                f"{_format_value(head['valid_r'])} | {_format_value(head['test_r'])} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/v0data"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    results = collate(args.checkpoint_root)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(results))
    if not args.json_output and not args.markdown_output:
        print(render_markdown(results), end="")


if __name__ == "__main__":
    main()
