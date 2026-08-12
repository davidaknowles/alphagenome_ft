import json
from pathlib import Path

import pytest

from scripts.v0data.compare_hda_rna_isolation import compare_isolation


def _write_run(root: Path, run: str, value: float) -> None:
    path = root / run / "metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "epoch": 1,
                "metrics": {
                    "valid": {
                        "hda_rna": {"differential_pearson_r": value},
                    }
                },
            }
        )
        + "\n"
    )


def _write_matrix(root: Path, values: dict[str, tuple[float, float]]) -> None:
    for strategy, (joint, isolated) in values.items():
        suffix = strategy.replace("+", "_")
        _write_run(
            root,
            f"hda-joint_{suffix}_neural_accessibility_bootstrap_screen",
            joint,
        )
        _write_run(
            root,
            f"hda-joint_{suffix}_rna_only_neural_accessibility_bootstrap_screen",
            isolated,
        )


def test_supports_isolation_when_both_strategies_improve(tmp_path: Path) -> None:
    _write_matrix(tmp_path, {"lora": (0.5, 0.55), "lora+locon": (0.52, 0.56)})

    result = compare_isolation(tmp_path)

    assert result["supports_modality_isolation"]
    assert result["mean_improvement"] == pytest.approx(0.045)


def test_rejects_isolation_when_one_strategy_regresses(tmp_path: Path) -> None:
    _write_matrix(tmp_path, {"lora": (0.5, 0.6), "lora+locon": (0.55, 0.54)})

    result = compare_isolation(tmp_path)

    assert result["mean_improvement"] > 0
    assert not result["passes_strategy_gate"]
    assert not result["supports_modality_isolation"]
