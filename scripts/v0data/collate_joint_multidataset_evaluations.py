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


def _resolve_manifest(path_value: str, *, config_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def expected_head_modalities(
    dataset_config: Path,
) -> dict[tuple[str, str], dict[str, str] | None]:
    """Return manifest-declared modality for every native-source head."""
    payload = json.loads(dataset_config.read_text())
    modalities: dict[tuple[str, str], dict[str, str] | None] = {}
    for dataset in payload.get("datasets", ()):
        dataset_name = str(dataset["name"])
        for source in dataset.get("sources", ()):
            route = (dataset_name, str(source["name"]))
            manifest_value = source.get("targets_config")
            if manifest_value is None:
                modalities[route] = None
                continue
            manifest_path = _resolve_manifest(
                str(manifest_value), config_path=dataset_config
            )
            manifest = json.loads(manifest_path.read_text())
            route_modalities: dict[str, str] = {}
            for head in manifest.get("heads", ()):
                kind = str(head.get("kind", "")).lower()
                if kind == "atac":
                    modality = "atac"
                elif kind == "rna_seq":
                    modality = "rna"
                else:
                    continue
                head_id = str(head["id"])
                if head_id in route_modalities:
                    raise ValueError(f"Duplicate head ID {head_id!r} in {manifest_path}.")
                route_modalities[head_id] = modality
            modalities[route] = route_modalities
    return modalities


def _fallback_head_modality(head: str) -> str | None:
    tokens = set(head.lower().split("_"))
    if "atac" in tokens:
        return "atac"
    if "rna" in tokens:
        return "rna"
    return None


def collate(
    dataset_config: Path,
    evaluation_root: Path,
    *,
    run_suffix: str = "",
    runs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if runs is None:
        runs = {
            f"{strategy_path}{run_suffix}": strategy_label
            for strategy_path, strategy_label in STRATEGIES.items()
        }
    if not runs or len(set(runs.values())) != len(runs):
        raise ValueError("Evaluation run paths and labels must be nonempty and unique.")
    routes = expected_routes(dataset_config)
    head_modalities = expected_head_modalities(dataset_config)
    rows = []
    missing = []
    for run_path, strategy_label in runs.items():
        for dataset, source in routes:
            route_path = f"{dataset}_{source}"
            evaluation_path = (
                evaluation_root
                / run_path
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
                manifest_modalities = head_modalities[(dataset, source)]
                modality = (
                    _fallback_head_modality(str(head))
                    if manifest_modalities is None
                    else manifest_modalities.get(str(head))
                )
                if modality is None:
                    raise ValueError(
                        f"Could not determine modality for {head!r} in {evaluation_path}."
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "source": source,
                        "strategy": strategy_label,
                        "source_epoch": source_epoch,
                        "source_global_step": source_global_step,
                        "head": str(head),
                        "modality": modality,
                        "valid_r": float(valid_r),
                        "test_r": float(test_r),
                    }
                )
    if missing:
        raise FileNotFoundError(
            "Missing joint native-source evaluations:\n" + "\n".join(missing)
        )
    summaries = []
    for strategy_label in runs.values():
        strategy_rows = [row for row in rows if row["strategy"] == strategy_label]
        modality_rows = {
            modality: [row for row in strategy_rows if row["modality"] == modality]
            for modality in ("atac", "rna")
        }

        def mean_or_none(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

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
                **{
                    f"mean_{modality}_{split}_r": mean_or_none(
                        [row[f"{split}_r"] for row in modality_rows[modality]]
                    )
                    for modality in ("atac", "rna")
                    for split in ("valid", "test")
                },
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
            "| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    def format_optional(value: Any) -> str:
        if isinstance(value, (int, float)) and math.isfinite(value):
            return f"{value:.4f}"
        return ""

    for summary in result["strategy_summaries"]:
        lines.append(
            f"| `{summary['strategy']}` | {summary['native_sources']} | "
            f"{summary['heads']} | {summary['mean_valid_r']:.4f} | "
            f"{summary['mean_test_r']:.4f} | {format_optional(summary['mean_atac_valid_r'])} | "
            f"{format_optional(summary['mean_atac_test_r'])} | "
            f"{format_optional(summary['mean_rna_valid_r'])} | "
            f"{format_optional(summary['mean_rna_test_r'])} |"
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
        "--run",
        action="append",
        default=[],
        metavar="PATH=LABEL",
        help="Evaluation directory and display label. Repeat for multiple selected runs.",
    )
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
    runs = None
    if args.run:
        if args.run_suffix:
            raise ValueError("--run-suffix cannot be combined with --run.")
        runs = {}
        for value in args.run:
            path, separator, label = value.partition("=")
            if not separator or not path or not label or path in runs:
                raise ValueError(f"Invalid or duplicate --run value: {value!r}")
            runs[path] = label
    result = collate(
        args.dataset_config,
        args.evaluation_root,
        run_suffix=args.run_suffix,
        runs=runs,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(result))
    print(args.markdown_output)


if __name__ == "__main__":
    main()
