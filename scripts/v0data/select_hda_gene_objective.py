#!/usr/bin/env python3
"""Select an HDA gene-only correlation objective from completed LoRA screens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCREENS = (
    {"index": 0, "weight": 0.0, "suffix": "0"},
    {"index": 1, "weight": 0.1, "suffix": "0p1"},
    {"index": 2, "weight": 1.0, "suffix": "1"},
    {"index": 3, "weight": 10.0, "suffix": "10"},
)


def _read_score(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "evaluation.json"
    record = json.loads(path.read_text())
    if record.get("source_epoch") != 1:
        raise ValueError(f"Expected an epoch-one reevaluation in {path}.")
    valid = record.get("metrics", {}).get("valid", {})
    values = {
        head: metrics.get("differential_pearson_r") for head, metrics in valid.items()
    }
    if set(values) != {"hda_atac", "hda_rna"} or any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values.values()
    ):
        raise ValueError(f"Missing finite HDA ATAC/RNA validation correlations in {path}.")
    return {
        "evaluation_path": str(path),
        "valid_r": values,
        "mean_valid_r": sum(values.values()) / len(values),
    }


def select_objective(
    checkpoint_root: Path,
    *,
    minimum_improvement: float = 0.0,
) -> dict[str, Any]:
    """Select the strongest nonzero weight only when it beats weight zero."""
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("minimum_improvement must be finite and non-negative.")
    screens = []
    for screen in SCREENS:
        run = f"hda-joint_lora_geneonly_corrw{screen['suffix']}_screen"
        screens.append(
            {
                **screen,
                "run": run,
                **_read_score(checkpoint_root / run),
            }
        )
    baseline = screens[0]
    candidate = max(screens[1:], key=lambda screen: screen["mean_valid_r"])
    improvement = candidate["mean_valid_r"] - baseline["mean_valid_r"]
    selected = candidate if improvement > minimum_improvement else None
    return {
        "selection_metric": "mean validation signed double-centered Pearson R",
        "minimum_improvement": minimum_improvement,
        "baseline": baseline,
        "best_nonzero": candidate,
        "improvement_over_baseline": improvement,
        "selected": selected,
        "status": "selected nonzero objective" if selected else "no nonzero improvement",
        "screens": screens,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# HDA gene-objective selection",
        "",
        "A nonzero correlation weight advances to LoRA plus LoCon only when its LoRA epoch-one mean validation signed double-centered Pearson correlation exceeds the gene-only weight-zero baseline.",
        "",
        "| Weight | ATAC validation R | RNA validation R | Mean validation R |",
        "|---:|---:|---:|---:|",
    ]
    for screen in result["screens"]:
        lines.append(
            f"| {screen['weight']:g} | {screen['valid_r']['hda_atac']:.4f} | "
            f"{screen['valid_r']['hda_rna']:.4f} | {screen['mean_valid_r']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Status, {result['status']}.",
            f"Improvement over weight zero, {result['improvement_over_baseline']:.4f}.",
        ]
    )
    if result["selected"]:
        lines.append(f"Selected weight, {result['selected']['weight']:g}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/v0data"))
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    result = select_objective(
        args.checkpoint_root,
        minimum_improvement=args.minimum_improvement,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(result))
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
