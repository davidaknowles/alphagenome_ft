import json

import pytest

from scripts.v0data.collate_joint_species_evaluations import collate, render_markdown


def test_collates_joint_species_evaluation(tmp_path):
    run = tmp_path / "zemke2023_mouse_lora_locon_joint_epoch1_eval"
    run.mkdir()
    (run / "evaluation.json").write_text(
        json.dumps(
            {
                "source_epoch": 1,
                "source_global_step": 9072,
                "metrics": {"test": {"zemke2023_rna": {"differential_pearson_r": 0.42}}},
            }
        )
    )

    results = collate(tmp_path)

    assert results["evaluations"][0]["strategy"] == "lora+locon"
    assert "| `mouse` | `lora+locon` | 1 | `test` | `zemke2023_rna` | 0.4200 |" in render_markdown(
        results
    )


def test_rejects_epoch_mismatch(tmp_path):
    run = tmp_path / "zemke2023_human_lora_joint_epoch1_eval"
    run.mkdir()
    (run / "evaluation.json").write_text(
        json.dumps({"source_epoch": 2, "source_global_step": 10, "metrics": {"test": {}}})
    )

    with pytest.raises(ValueError, match="names epoch 1"):
        collate(tmp_path)
