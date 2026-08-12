import json
from pathlib import Path

import pytest

from scripts.v0data.prepare_head_correlation_config import prepare_config


def _manifest(path: Path, *, gene_supervision: bool = True) -> Path:
    rna = {
        "id": "rna",
        "kind": "rna_seq",
        "targets": [],
        "double_centered_correlation_loss_weight": 10.0,
    }
    if gene_supervision:
        rna["gene_supervision"] = {
            "path": "genes.npz",
            "loss_weight": 1.0,
            "coverage_loss_weight": 1.0,
        }
    path.write_text(json.dumps({"dataset": "test", "heads": [rna]}))
    return path


def test_prepare_config_changes_only_correlation_weights(tmp_path: Path) -> None:
    result = prepare_config(
        _manifest(tmp_path / "targets.json"),
        head_id="rna",
        double_centered_weight=0.0,
        row_centered_weight=1.0,
    )
    head = result["heads"][0]
    assert head["double_centered_correlation_loss_weight"] == 0.0
    assert head["row_centered_correlation_loss_weight"] == 1.0
    assert head["gene_supervision"]["coverage_loss_weight"] == 1.0


def test_prepare_config_requires_gene_supervision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no direct gene supervision"):
        prepare_config(
            _manifest(tmp_path / "targets.json", gene_supervision=False),
            head_id="rna",
            double_centered_weight=0.0,
            row_centered_weight=1.0,
        )
