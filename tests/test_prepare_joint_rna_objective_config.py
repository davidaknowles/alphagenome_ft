import json
from pathlib import Path

from scripts.v0data.prepare_joint_rna_objective_config import prepare_config


def test_prepare_config_updates_only_rna_objectives(tmp_path: Path) -> None:
    target_path = tmp_path / "source" / "targets.json"
    target_path.parent.mkdir()
    target_path.write_text(
        json.dumps(
            {
                "heads": [
                    {"id": "study_atac", "loss_weight": 1.0},
                    {
                        "id": "study_rna",
                        "loss_weight": 1.0,
                        "double_centered_correlation_loss_weight": 0.1,
                    },
                ]
            }
        )
    )
    dataset_path = tmp_path / "source" / "datasets.json"
    source = {
        "datasets": [
            {
                "name": "study",
                "sources": [{"name": "human", "targets_config": "targets.json"}],
            }
        ]
    }

    result = prepare_config(
        source,
        source_path=dataset_path,
        output_dir=tmp_path / "output",
        loss_weight=2.0,
        correlation_loss_weight=None,
    )

    copied_path = Path(result["datasets"][0]["sources"][0]["targets_config"])
    copied = json.loads(copied_path.read_text())
    assert copied["heads"][0] == {"id": "study_atac", "loss_weight": 1.0}
    assert copied["heads"][1]["loss_weight"] == 2.0
    assert copied["heads"][1]["double_centered_correlation_loss_weight"] == 0.1
    assert result["objective_variant"]["updated_heads"] == ["study_rna"]


def test_prepare_config_can_replace_rna_correlation_weight(tmp_path: Path) -> None:
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps({"heads": [{"id": "study_rna"}]}))
    source = {
        "datasets": [
            {
                "name": "study",
                "sources": [{"name": "human", "targets_config": str(target_path)}],
            }
        ]
    }

    result = prepare_config(
        source,
        source_path=tmp_path / "datasets.json",
        output_dir=tmp_path / "output",
        loss_weight=None,
        correlation_loss_weight=3.0,
    )

    copied_path = Path(result["datasets"][0]["sources"][0]["targets_config"])
    copied = json.loads(copied_path.read_text())
    assert copied["heads"][0]["double_centered_correlation_loss_weight"] == 3.0
