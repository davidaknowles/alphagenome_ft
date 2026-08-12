#!/usr/bin/env python3
"""Audit canonical validation correlations against modality targets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v0data.audit_adapter_coverage import EXPECTED_DATASETS
from scripts.v0data.collate_adapter_comparisons import collate

ATAC_ONLY_DATASETS = frozenset({"hda"})


def _modality(head: str) -> str | None:
    lowered = head.lower()
    if "atac" in lowered:
        return "ATAC"
    if "rna" in lowered:
        return "RNA"
    return None


def audit_metric_targets(
    canonical_results: dict[str, Any],
    *,
    threshold: float = 0.8,
    expected_datasets: tuple[str, ...] = EXPECTED_DATASETS,
) -> dict[str, Any]:
    """Report the best strategy-selected checkpoint for each expected modality."""
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in canonical_results.get("runs", ()):
        for head in run.get("heads", ()):
            modality = _modality(str(head.get("head", "")))
            valid_r = head.get("valid_r")
            if modality is None or not isinstance(valid_r, (int, float)) or not math.isfinite(valid_r):
                continue
            candidates.setdefault((run["dataset"], modality), []).append(
                {
                    "strategy": run["strategy"],
                    "selected_epoch": run.get("selected_epoch"),
                    "head": head["head"],
                    "valid_r": valid_r,
                    "test_r": head.get("test_r"),
                }
            )

    rows = []
    for dataset in expected_datasets:
        modalities = ("ATAC",) if dataset in ATAC_ONLY_DATASETS else ("ATAC", "RNA")
        for modality in modalities:
            available = candidates.get((dataset, modality), ())
            best = max(available, key=lambda row: row["valid_r"], default=None)
            rows.append(
                {
                    "dataset": dataset,
                    "modality": modality,
                    "threshold": threshold,
                    "status": (
                        "missing evidence"
                        if best is None
                        else "target reached"
                        if best["valid_r"] >= threshold
                        else "below target"
                    ),
                    "gap_to_target": (
                        None if best is None else max(0.0, threshold - best["valid_r"])
                    ),
                    **(best or {}),
                }
            )
    return {"metric": "validation signed double-centered Pearson R", "rows": rows}


def _format(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) and math.isfinite(value) else ""


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Canonical metric-target audit",
        "",
        "The table reports the stronger of the two strategy-selected canonical checkpoints for each modality. Each strategy checkpoint is selected by mean validation signed double-centered Pearson correlation across its heads. Missing evidence is distinct from a measured value below the target.",
        "",
        "| Dataset | Modality | Strategy | Epoch | Validation R | Test R | Gap to 0.8 | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{row['dataset']}` | {row['modality']} | "
            f"{('`' + row['strategy'] + '`') if row.get('strategy') else ''} | "
            f"{row.get('selected_epoch', '')} | {_format(row.get('valid_r'))} | "
            f"{_format(row.get('test_r'))} | {_format(row.get('gap_to_target'))} | "
            f"{row['status']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/v0data"))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    result = audit_metric_targets(collate(args.checkpoint_root), threshold=args.threshold)
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
