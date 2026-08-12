#!/usr/bin/env python
"""Collate native-source evaluations from joint multi-dataset checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


STRATEGIES = {"lora": "lora", "lora_locon": "lora+locon"}


def expected_routes(dataset_config: Path) -> list[tuple[str, str]]:
    payload = json.loads(dataset_config.read_text())
    routes = []
    for dataset in payload.get("datasets", ()):
        dataset_name = str(dataset["name"])
        for source in dataset.get("sources", ()):
            routes.append((dataset_name, str(source["name"])))
    if not routes:
        raise ValueError(f"Dataset configuration contains no native sources: {dataset_config}")
    return routes


def collate(
    dataset_config: Path,
    evaluation_root: Path,
    *,
    run_suffix: str = "",
) -> dict[str, Any]:
    rows = []
    missing = []
    for strategy_path, strategy_label in STRATEGIES.items():
        for dataset, source in expected_routes(dataset_config):
            route_path = f"{dataset}_{source}"
            evaluation_path = (
                evaluation_root
                / f"{strategy_path}{run_suffix}"
                / route_path
                / "evaluation.json"
            )
            if not evaluation_path.exists():
                missing.append(str(evaluation_path))
                continue
            payload = json.loads(evaluation_path.read_text())
            source_epoch = payload.get("source_epoch")
            source_global_step = payload.get("source_global_step")
            if not isinstance(source_epoch, int) or source_epoch < 1:
                raise ValueError(f"Invalid source epoch in {evaluation_path}: {source_epoch!r}")
            if not isinstance(source_global_step, int) or source_global_step < 1:
                raise ValueError(
                    f"Invalid source global step in {evaluation_path}: {source_global_step!r}"
                )
            metrics = payload.get("metrics", {})
            valid_heads = metrics.get("valid", {})
            test_heads = metrics.get("test", {})
            if not valid_heads or set(valid_heads) != set(test_heads):
                raise ValueError(
                    f"Validation and test heads do not match in {evaluation_path}."
                )
            for head, valid_metrics in valid_heads.items():
                valid_r = valid_metrics.get("differential_pearson_r")
                test_r = test_heads[head].get("differential_pearson_r")
                if not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (valid_r, test_r)
                ):
                    raise ValueError(
                        f"Non-finite differential correlation for {head} in {evaluation_path}."
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "source": source,
                        "strategy": strategy_label,
                        "source_epoch": source_epoch,
                        "source_global_step": source_global_step,
                        "head": str(head),
                        "valid_r": float(valid_r),
                        "test_r": float(test_r),
                    }
                )
    if missing:
        raise FileNotFoundError(
            "Missing joint native-source evaluations:\n" + "\n".join(missing)
        )
    summaries = []
    for strategy_label in STRATEGIES.values():
        strategy_rows = [row for row in rows if row["strategy"] == strategy_label]
        summaries.append(
            {
                "strategy": strategy_label,
                "native_sources": len(
                    {(row["dataset"], row["source"]) for row in strategy_rows}
                ),
                "heads": len(strategy_rows),
                "mean_valid_r": sum(row["valid_r"] for row in strategy_rows)
                / len(strategy_rows),
                "mean_test_r": sum(row["test_r"] for row in strategy_rows)
                / len(strategy_rows),
            }
        )
    return {"rows": rows, "strategy_summaries": summaries}


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Joint native-source evaluation",
        "",
        "Each row evaluates one head from the same union model on one native source. "
        "R is signed double-centered Pearson correlation.",
        "",
        "| Dataset | Source | Strategy | Epoch | Head | Validation R | Test R |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{row['dataset']}` | `{row['source']}` | `{row['strategy']}` | "
            f"{row['source_epoch']} | `{row['head']}` | {row['valid_r']:.4f} | "
            f"{row['test_r']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Strategy | Native sources | Heads | Mean validation R | Mean test R |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for summary in result["strategy_summaries"]:
        lines.append(
            f"| `{summary['strategy']}` | {summary['native_sources']} | "
            f"{summary['heads']} | {summary['mean_valid_r']:.4f} | "
            f"{summary['mean_test_r']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("outputs/v0data/joint-all-nonencode/datasets.json"),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("evaluations/v0data/joint_all_nonencode"),
    )
    parser.add_argument("--run-suffix", default="")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/v0data_joint_multidataset_evaluations.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("results/v0data_joint_multidataset_evaluations.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collate(
        args.dataset_config,
        args.evaluation_root,
        run_suffix=args.run_suffix,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(result))
    print(args.markdown_output)


if __name__ == "__main__":
    main()
