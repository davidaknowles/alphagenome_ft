#!/usr/bin/env python3
"""Compare matched joint and RNA-only Mannens neural-initialized screens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


STRATEGIES = ("lora", "lora+locon")


def _read_rna(path: Path) -> float:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches = [record for record in records if record.get("epoch") == 1]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one epoch-one record in {path}.")
    value = (
        matches[0]
        .get("metrics", {})
        .get("valid", {})
        .get("hda_rna", {})
        .get("differential_pearson_r")
    )
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing finite HDA RNA validation R in {path}.")
    return float(value)


def compare_isolation(
    checkpoint_root: Path,
    *,
    minimum_improvement: float = 0.0,
    maximum_strategy_regression: float = 0.0,
) -> dict[str, Any]:
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("minimum_improvement must be finite and non-negative.")
    if not math.isfinite(maximum_strategy_regression) or maximum_strategy_regression < 0:
        raise ValueError("maximum_strategy_regression must be finite and non-negative.")
    rows = []
    for strategy in STRATEGIES:
        suffix = strategy.replace("+", "_")
        joint_run = f"hda-joint_{suffix}_neural_accessibility_bootstrap_screen"
        isolated_run = (
            f"hda-joint_{suffix}_rna_only_neural_accessibility_bootstrap_screen"
        )
        joint_path = checkpoint_root / joint_run / "metrics.jsonl"
        isolated_path = checkpoint_root / isolated_run / "metrics.jsonl"
        joint_r = _read_rna(joint_path)
        isolated_r = _read_rna(isolated_path)
        rows.append(
            {
                "strategy": strategy,
                "joint_run": joint_run,
                "rna_only_run": isolated_run,
                "joint_valid_r": joint_r,
                "rna_only_valid_r": isolated_r,
                "improvement": isolated_r - joint_r,
            }
        )
    mean_improvement = sum(row["improvement"] for row in rows) / len(rows)
    passes_strategy_gate = all(
        row["improvement"] >= -maximum_strategy_regression for row in rows
    )
    supports_isolation = (
        mean_improvement > minimum_improvement and passes_strategy_gate
    )
    return {
        "metric": "epoch-one validation signed double-centered Pearson R for HDA RNA",
        "minimum_improvement": minimum_improvement,
        "maximum_strategy_regression": maximum_strategy_regression,
        "mean_improvement": mean_improvement,
        "passes_strategy_gate": passes_strategy_gate,
        "supports_modality_isolation": supports_isolation,
        "status": (
            "RNA-only adaptation improved both-strategy evidence"
            if supports_isolation
            else "RNA-only adaptation did not advance"
        ),
        "strategies": rows,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# HDA RNA modality-isolation comparison",
        "",
        "Joint and RNA-only runs use the same deterministic pretrained RNA parameters, objective, windows, optimizer settings, and epoch. A positive difference therefore tests whether removing the ATAC gradient improves RNA adaptation.",
        "",
        "| Strategy | Joint RNA R | RNA-only R | Difference |",
        "|---|---:|---:|---:|",
    ]
    for row in result["strategies"]:
        lines.append(
            f"| `{row['strategy']}` | {row['joint_valid_r']:.4f} | "
            f"{row['rna_only_valid_r']:.4f} | {row['improvement']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Mean difference, {result['mean_improvement']:.4f}.",
            f"Status, {result['status']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/v0data"))
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--maximum-strategy-regression", type=float, default=0.0)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    result = compare_isolation(
        args.checkpoint_root,
        minimum_improvement=args.minimum_improvement,
        maximum_strategy_regression=args.maximum_strategy_regression,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(result))
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
