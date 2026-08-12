#!/usr/bin/env python3
"""Select a pretrained-head initializer from matched HDA and Liu screens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DATASETS = (
    {
        "dataset": "hda-joint",
        "heads": {"ATAC": "hda_atac", "RNA": "hda_rna"},
        "canonical": {
            "lora": "hda-joint_lora",
            "lora+locon": "hda-joint_lora_locon",
        },
    },
    {
        "dataset": "liu-hdma",
        "heads": {"ATAC": "liu_atac", "RNA": "liu_rna"},
        "canonical": {
            "lora": "liu-hdma_lora_geneonly_corrw1",
            "lora+locon": "liu-hdma_lora_locon_geneonly_corrw1",
        },
    },
)
INITIALIZERS = (
    {"name": "none", "suffix": None},
    {"name": "bootstrap", "suffix": "bootstrap_screen"},
    {"name": "neural_bootstrap", "suffix": "neural_bootstrap_screen"},
    {
        "name": "neural_accessibility_bootstrap",
        "suffix": "neural_accessibility_bootstrap_screen",
    },
)
STRATEGIES = ("lora", "lora+locon")


def _epoch_one(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches = [record for record in records if record.get("epoch") == 1]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one epoch-one record in {path}.")
    return matches[0]


def _run_name(dataset: dict[str, Any], initializer: dict[str, Any], strategy: str) -> str:
    if initializer["suffix"] is None:
        return dataset["canonical"][strategy]
    strategy_suffix = strategy.replace("+", "_")
    return f"{dataset['dataset']}_{strategy_suffix}_{initializer['suffix']}"


def select_initializer(
    checkpoint_root: Path,
    *,
    minimum_improvement: float = 0.0,
    maximum_modality_regression: float = 0.0,
) -> dict[str, Any]:
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("minimum_improvement must be finite and non-negative.")
    if not math.isfinite(maximum_modality_regression) or maximum_modality_regression < 0:
        raise ValueError("maximum_modality_regression must be finite and non-negative.")

    initializers = []
    for initializer in INITIALIZERS:
        arms = []
        for dataset in DATASETS:
            for strategy in STRATEGIES:
                run = _run_name(dataset, initializer, strategy)
                path = checkpoint_root / run / "metrics.jsonl"
                record = _epoch_one(path)
                valid = record.get("metrics", {}).get("valid", {})
                modality_values = {
                    modality: valid.get(head, {}).get("differential_pearson_r")
                    for modality, head in dataset["heads"].items()
                }
                if any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in modality_values.values()
                ):
                    raise ValueError(f"Missing finite ATAC/RNA validation R in {path}.")
                arms.append(
                    {
                        "dataset": dataset["dataset"],
                        "strategy": strategy,
                        "run": run,
                        "metrics_path": str(path),
                        "valid_r": modality_values,
                        "mean_valid_r": sum(modality_values.values()) / 2,
                    }
                )
        modality_means = {
            modality: sum(arm["valid_r"][modality] for arm in arms) / len(arms)
            for modality in ("ATAC", "RNA")
        }
        initializers.append(
            {
                **initializer,
                "arms": arms,
                "modality_mean_valid_r": modality_means,
                "mean_valid_r": sum(modality_means.values()) / 2,
            }
        )

    baseline = initializers[0]
    candidate = max(initializers[1:], key=lambda item: item["mean_valid_r"])
    improvement = candidate["mean_valid_r"] - baseline["mean_valid_r"]
    regressions = {
        modality: baseline["modality_mean_valid_r"][modality]
        - candidate["modality_mean_valid_r"][modality]
        for modality in ("ATAC", "RNA")
    }
    passes_modality_gate = all(
        regression <= maximum_modality_regression for regression in regressions.values()
    )
    selected = (
        candidate
        if improvement > minimum_improvement and passes_modality_gate
        else None
    )
    return {
        "selection_metric": "mean epoch-one validation signed double-centered Pearson R across datasets, strategies, and modalities",
        "minimum_improvement": minimum_improvement,
        "maximum_modality_regression": maximum_modality_regression,
        "baseline": baseline,
        "best_candidate": candidate,
        "improvement_over_baseline": improvement,
        "modality_regression": regressions,
        "passes_modality_gate": passes_modality_gate,
        "selected": selected,
        "status": "selected initializer" if selected else "no initializer advanced",
        "initializers": initializers,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Pretrained-head initialization selection",
        "",
        "The selector compares epoch-one validation signed double-centered Pearson correlation under both adapter strategies on Mannens HDA and Liu HDMA. A candidate advances only if its overall mean improves and neither modality exceeds the allowed average regression.",
        "",
        "| Initializer | Dataset | Strategy | ATAC R | RNA R | Mean R |",
        "|---|---|---|---:|---:|---:|",
    ]
    for initializer in result["initializers"]:
        for arm in initializer["arms"]:
            lines.append(
                f"| `{initializer['name']}` | `{arm['dataset']}` | `{arm['strategy']}` | "
                f"{arm['valid_r']['ATAC']:.4f} | {arm['valid_r']['RNA']:.4f} | "
                f"{arm['mean_valid_r']:.4f} |"
            )
    lines.extend(
        [
            "",
            "| Initializer | ATAC mean R | RNA mean R | Overall mean R |",
            "|---|---:|---:|---:|",
        ]
    )
    for initializer in result["initializers"]:
        lines.append(
            f"| `{initializer['name']}` | "
            f"{initializer['modality_mean_valid_r']['ATAC']:.4f} | "
            f"{initializer['modality_mean_valid_r']['RNA']:.4f} | "
            f"{initializer['mean_valid_r']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Status, {result['status']}.",
            f"Improvement over random initialization, {result['improvement_over_baseline']:.4f}.",
        ]
    )
    if result["selected"]:
        lines.append(f"Selected initializer, `{result['selected']['name']}`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/v0data"))
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--maximum-modality-regression", type=float, default=0.0)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    result = select_initializer(
        args.checkpoint_root,
        minimum_improvement=args.minimum_improvement,
        maximum_modality_regression=args.maximum_modality_regression,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(result))
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
