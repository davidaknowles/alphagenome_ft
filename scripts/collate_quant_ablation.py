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


def _sort_batch(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
        if metrics_path.parent.name.startswith("smoke_"):
            continue
        payload = _load(metrics_path)
        if not payload:
            continue
        strategy = payload.get("strategy", metrics_path.parent.name)
        backend = payload.get("backend", "")
        gpu = payload.get("gpu", {})
        torch_cuda = payload.get("torch_cuda_memory") or {}
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
                        "batch_size": payload.get("batch_size"),
                        "examples": values.get("examples"),
                        "examples_per_sec": payload.get("examples_per_sec"),
                        "converted": quant.get("converted"),
                        "stdconv_effective": quant.get("standardized_convs_materialized"),
                        "elapsed": payload.get("elapsed_sec"),
                        "avg_gpu": gpu.get("avg_util_pct"),
                        "max_mem": gpu.get("max_mem_mib"),
                        "torch_max_alloc": torch_cuda.get("max_allocated_mib"),
                        "torch_max_reserved": torch_cuda.get("max_reserved_mib"),
                    }
                )

    rows.sort(
        key=lambda row: (
            row["backend"],
            row["strategy"],
            _sort_batch(row["batch_size"]),
            row["split"],
        )
    )

    output = args.output.expanduser().resolve() if args.output else root / "quant_ablation_results.md"
    lines = [
        "# Quantization Ablation Results",
        "",
        f"Root: `{root}`",
        "",
        "| backend | strategy | batch | split | loss | diff Pearson | r2_global | r2_loci | examples/s | examples | converted | stdconv eff | sec | avg GPU % | nvidia-smi max MiB | torch max alloc MiB | torch max reserved MiB |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {backend} | {strategy} | {batch_size} | {split} | {loss} | {diff} | {r2_global} | {r2_loci} | {examples_per_sec} | {examples} | {converted} | {stdconv_effective} | {elapsed} | {avg_gpu} | {max_mem} | {torch_max_alloc} | {torch_max_reserved} |".format(
                backend=row["backend"],
                strategy=row["strategy"],
                batch_size=row["batch_size"],
                split=row["split"],
                loss=_fmt(row["loss"]),
                diff=_fmt(row["diff"]),
                r2_global=_fmt(row["r2_global"]),
                r2_loci=_fmt(row["r2_loci"]),
                examples_per_sec=_fmt(row["examples_per_sec"], 2),
                examples=row["examples"],
                converted=row["converted"],
                stdconv_effective=row["stdconv_effective"],
                elapsed=_fmt(row["elapsed"], 1),
                avg_gpu=_fmt(row["avg_gpu"], 1),
                max_mem=_fmt(row["max_mem"], 0),
                torch_max_alloc=_fmt(row["torch_max_alloc"], 0),
                torch_max_reserved=_fmt(row["torch_max_reserved"], 0),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Quantized rows use implemented Torch module replacements; old JAX NF4/FP8 roundtrip simulations are no longer scheduled.",
            "- LoCon folding into standardized convs is approximate; inspect `merged*/merge_metadata.json` and default merged metrics first.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
