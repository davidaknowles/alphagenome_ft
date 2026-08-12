#!/usr/bin/env python3
"""Collate persisted first-batch per-head gradient diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def collate(checkpoint_root: Path) -> dict[str, Any]:
    runs = []
    for path in sorted(checkpoint_root.glob("*/gradient_diagnostics.json")):
        record = json.loads(path.read_text())
        heads = record.get("heads")
        cosines = record.get("adapter_gradient_cosines", {})
        if not isinstance(heads, dict) or not heads:
            raise ValueError(f"{path} does not contain per-head diagnostics.")
        runs.append(
            {
                "run": path.parent.name,
                "epoch": record.get("epoch"),
                "global_step_before_update": record.get("global_step_before_update"),
                "heads": heads,
                "adapter_gradient_cosines": cosines,
                "path": str(path),
            }
        )
    return {"runs": runs}


def _format(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Per-head gradient diagnostics",
        "",
        "Each row measures one head on the same first training batch before the optimizer update. Adapter norms use shared LoRA and LoCon parameters; weighted norms include the configured outer head weight.",
        "",
        "| Run | Head | Loss | Adapter norm | Weighted adapter norm | Head norm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in results["runs"]:
        for head_name, values in sorted(run["heads"].items()):
            lines.append(
                f"| `{run['run']}` | `{head_name}` | {_format(values.get('loss'))} | "
                f"{_format(values.get('adapter_gradient_norm'))} | "
                f"{_format(values.get('weighted_adapter_gradient_norm'))} | "
                f"{_format(values.get('head_gradient_norm'))} |"
            )
    lines.extend(
        [
            "",
            "| Run | Head pair | Adapter-gradient cosine |",
            "|---|---|---:|",
        ]
    )
    for run in results["runs"]:
        for pair, value in sorted(run["adapter_gradient_cosines"].items()):
            lines.append(f"| `{run['run']}` | `{pair}` | {_format(value)} |")
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
