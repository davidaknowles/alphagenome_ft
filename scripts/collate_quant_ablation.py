#!/usr/bin/env python
"""Collate quantization ablation metrics into a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        payload = _load(metrics_path)
        if not payload:
            continue
        strategy = payload.get("strategy", metrics_path.parent.name)
        backend = payload.get("backend", "")
        gpu = payload.get("gpu", {})
        quant = payload.get("quantization", {})
        for split, heads in payload.get("splits", {}).items():
            for head, values in heads.items():
                rows.append(
                    {
                        "backend": backend,
                        "strategy": strategy,
                        "split": split,
                        "head": head,
                        "loss": values.get("loss"),
                        "diff": values.get("differential_pearson_r"),
                        "r2_global": values.get("r2_global"),
                        "r2_loci": values.get("r2_over_loci"),
                        "batches": values.get("batches"),
                        "converted": quant.get("converted"),
                        "simulated": quant.get("simulated_storage"),
                        "elapsed": payload.get("elapsed_sec"),
                        "avg_gpu": gpu.get("avg_util_pct"),
                        "max_mem": gpu.get("max_mem_mib"),
                    }
                )

    output = args.output.expanduser().resolve() if args.output else root / "quant_ablation_results.md"
    lines = [
        "# Quantization Ablation Results",
        "",
        f"Root: `{root}`",
        "",
        "| backend | strategy | split | loss | diff Pearson | r2_global | r2_loci | converted | simulated | sec | avg GPU % | max MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {backend} | {strategy} | {split} | {loss} | {diff} | {r2_global} | {r2_loci} | {converted} | {simulated} | {elapsed} | {avg_gpu} | {max_mem} |".format(
                backend=row["backend"],
                strategy=row["strategy"],
                split=row["split"],
                loss=_fmt(row["loss"]),
                diff=_fmt(row["diff"]),
                r2_global=_fmt(row["r2_global"]),
                r2_loci=_fmt(row["r2_loci"]),
                converted=row["converted"],
                simulated=row["simulated"],
                elapsed=_fmt(row["elapsed"], 1),
                avg_gpu=_fmt(row["avg_gpu"], 1),
                max_mem=_fmt(row["max_mem"], 0),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- JAX NF4/FP8 policies are eval-time roundtrip simulations unless `simulated=false`.",
            "- LoCon folding into standardized convs is approximate; inspect `merged*/merge_metadata.json` and default merged metrics first.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
