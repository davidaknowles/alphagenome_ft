import json
from pathlib import Path

import pytest

from scripts.v0data.select_hda_gene_objective import SCREENS, select_objective


def _write_screen(root: Path, suffix: str, atac: float, rna: float) -> None:
    run = root / f"hda-joint_lora_geneonly_corrw{suffix}_screen"
    run.mkdir()
    record = {
        "source_epoch": 1,
        "metrics": {
            "valid": {
                "hda_atac": {"differential_pearson_r": atac},
                "hda_rna": {"differential_pearson_r": rna},
            }
        },
    }
    (run / "evaluation.json").write_text(json.dumps(record) + "\n")


def test_rejects_training_metric_without_reevaluation(tmp_path: Path) -> None:
    for screen in SCREENS:
        run = tmp_path / f"hda-joint_lora_geneonly_corrw{screen['suffix']}_screen"
        run.mkdir()
        (run / "metrics.jsonl").write_text("{}\n")

    with pytest.raises(FileNotFoundError):
        select_objective(tmp_path)


def test_selects_best_nonzero_weight_only_when_mean_improves(tmp_path: Path) -> None:
    values = [(0.7, 0.5), (0.7, 0.55), (0.69, 0.58), (0.60, 0.60)]
    for screen, (atac, rna) in zip(SCREENS, values, strict=True):
        _write_screen(tmp_path, screen["suffix"], atac, rna)

    result = select_objective(tmp_path)

    assert result["selected"]["weight"] == 1.0
    assert result["improvement_over_baseline"] == pytest.approx(0.035)


def test_does_not_advance_nonzero_weight_without_improvement(tmp_path: Path) -> None:
    values = [(0.7, 0.5), (0.69, 0.50), (0.68, 0.51), (0.60, 0.59)]
    for screen, (atac, rna) in zip(SCREENS, values, strict=True):
        _write_screen(tmp_path, screen["suffix"], atac, rna)

    result = select_objective(tmp_path)

    assert result["selected"] is None
    assert result["status"] == "no nonzero improvement"
