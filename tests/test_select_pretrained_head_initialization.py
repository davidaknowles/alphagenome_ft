import json
from pathlib import Path

import pytest

from scripts.v0data.select_pretrained_head_initialization import (
    DATASETS,
    INITIALIZERS,
    STRATEGIES,
    _run_name,
    select_initializer,
)


def _write_inputs(root: Path, scores: dict[str, tuple[float, float]]) -> None:
    for initializer in INITIALIZERS:
        for dataset in DATASETS:
            for strategy in STRATEGIES:
                run = _run_name(dataset, initializer, strategy)
                path = root / run / "metrics.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                atac, rna = scores[initializer["name"]]
                heads = dataset["heads"]
                record = {
                    "epoch": 1,
                    "metrics": {
                        "valid": {
                            heads["ATAC"]: {"differential_pearson_r": atac},
                            heads["RNA"]: {"differential_pearson_r": rna},
                        }
                    },
                }
                path.write_text(json.dumps(record) + "\n")


def test_selects_initializer_that_improves_both_modalities(tmp_path: Path) -> None:
    _write_inputs(
        tmp_path,
        {
            "none": (0.70, 0.50),
            "bootstrap": (0.71, 0.51),
            "neural_bootstrap": (0.72, 0.53),
            "neural_accessibility_bootstrap": (0.74, 0.55),
        },
    )

    result = select_initializer(tmp_path)

    assert result["selected"]["name"] == "neural_accessibility_bootstrap"
    assert result["improvement_over_baseline"] == pytest.approx(0.045)


def test_rejects_best_mean_when_one_modality_regresses(tmp_path: Path) -> None:
    _write_inputs(
        tmp_path,
        {
            "none": (0.70, 0.50),
            "bootstrap": (0.69, 0.60),
            "neural_bootstrap": (0.68, 0.59),
            "neural_accessibility_bootstrap": (0.67, 0.58),
        },
    )

    result = select_initializer(tmp_path)

    assert result["best_candidate"]["name"] == "bootstrap"
    assert result["selected"] is None
    assert not result["passes_modality_gate"]
