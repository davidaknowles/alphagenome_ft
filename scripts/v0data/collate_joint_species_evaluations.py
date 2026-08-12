#!/usr/bin/env python3
"""Collate native-species evaluations of joint cross-species checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

ZEMKE_RUN_PATTERN = re.compile(
    r"^zemke2023_(?P<species>human|macaque|marmoset|mouse)_"
    r"(?P<strategy>lora|lora_locon)_joint_epoch(?P<epoch>\d+)_eval$"
)
JOHANSEN_RUN_PATTERN = re.compile(
    r"^johansen_joint_(?P<strategy>lora|lora_locon)_"
    r"(?P<species>human|macaque|marmoset)_eval$"
)


def collate(checkpoint_root: Path) -> dict[str, Any]:
    evaluations = []
    for path in sorted(checkpoint_root.glob("*/evaluation.json")):
        zemke_match = ZEMKE_RUN_PATTERN.fullmatch(path.parent.name)
        johansen_match = JOHANSEN_RUN_PATTERN.fullmatch(path.parent.name)
        match = zemke_match or johansen_match
        if match is None:
            continue
        record = json.loads(path.read_text())
        source_epoch = record.get("source_epoch")
        if not isinstance(source_epoch, int) or source_epoch < 1:
            raise ValueError(f"{path} does not record a positive integer source epoch.")
        if zemke_match is not None:
            expected_epoch = int(zemke_match.group("epoch"))
            if source_epoch != expected_epoch:
                raise ValueError(
                    f"{path} names epoch {expected_epoch} but records source epoch "
                    f"{source_epoch}."
                )
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"{path} does not contain nonempty split metrics.")
        evaluations.append(
            {
                "dataset": "zemke2023_joint" if zemke_match else "johansen_joint",
                "species": match.group("species"),
                "strategy": match.group("strategy").replace("_", "+"),
                "source_epoch": source_epoch,
                "source_global_step": record.get("source_global_step"),
                "metrics": metrics,
                "evaluation_path": str(path),
            }
        )
    return {"evaluations": evaluations}


def _format(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return ""
    return f"{value:.4f}"


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Joint-checkpoint native-species evaluations",
        "",
        "Each joint cross-species checkpoint is evaluated without parameter updates against the native reference and target manifest for each species.",
        "",
        "| Dataset | Species | Strategy | Epoch | Split | Head | Differential R |",
        "|---|---|---|---:|---|---|---:|",
    ]
    for evaluation in results["evaluations"]:
        for split, heads in evaluation["metrics"].items():
            for head_name, metrics in heads.items():
                lines.append(
                    f"| `{evaluation['dataset']}` | `{evaluation['species']}` | "
                    f"`{evaluation['strategy']}` | "
                    f"{evaluation['source_epoch']} | `{split}` | `{head_name}` | "
                    f"{_format(metrics.get('differential_pearson_r'))} |"
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
