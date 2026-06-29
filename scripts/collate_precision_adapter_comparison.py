#!/usr/bin/env python
"""Collate per-cell precision/adapter benchmark summaries into markdown."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _metric(row: dict, *names: str):
    for name in names:
        if name in row:
            return row[name]
    return None


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    summaries = sorted(output_root.glob("*/summary.json"))
    rows = [json.loads(path.read_text()) for path in summaries]
    rows.sort(key=lambda r: (r.get("backend", ""), r.get("precision", ""), r.get("adapter_strategy", "")))

    lines = [
        "# Precision/Adapter Backend Comparison",
        "",
        f"Output root: `{output_root}`",
        "",
        "| Backend | Precision | Strategy | Status | Wall sec | Avg GPU % | Max VRAM MiB | Train loss | Valid loss | Test loss | Valid R2 global | Test R2 global | Differential Pearson R |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        differential_pearson = _metric(
            row,
            "valid_differential_pearson_r",
            "test_differential_pearson_r",
            "atac_128bp_differential_pearson_r",
            "val_loss_atac_128bp_differential_pearson_r",
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("backend", "NA")),
                    str(row.get("precision", "NA")),
                    str(row.get("adapter_strategy", "NA")),
                    str(row.get("status", "NA")),
                    _fmt(row.get("elapsed_sec"), 2),
                    _fmt(row.get("avg_gpu_util_pct"), 2),
                    _fmt(row.get("max_mem_mib"), 0),
                    _fmt(row.get("train_loss"), 4),
                    _fmt(_metric(row, "valid_loss", "val_loss_atac_loss"), 4),
                    _fmt(row.get("test_loss"), 4),
                    _fmt(row.get("valid_r2_global"), 4),
                    _fmt(row.get("test_r2_global"), 4),
                    _fmt(differential_pearson, 4),
                ]
            )
            + " |"
        )
    text = "\n".join(lines) + "\n"
    output_md = args.output_md or (output_root / "comparison.md")
    output_md.write_text(text)
    print(output_md)


if __name__ == "__main__":
    main()
