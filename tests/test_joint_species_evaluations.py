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
    assert "| `zemke2023_joint` | `mouse` | `lora+locon` | 1 | `test` | `zemke2023_rna` | 0.4200 |" in render_markdown(
        results
    )


def test_collates_johansen_native_species_evaluation(tmp_path):
    run = tmp_path / "johansen_joint_lora_macaque_eval"
    run.mkdir()
    (run / "evaluation.json").write_text(
        json.dumps(
            {
                "source_epoch": 3,
                "source_global_step": 43587,
                "metrics": {"valid": {"allen_atac": {"differential_pearson_r": 0.5}}},
            }
        )
    )

    results = collate(tmp_path)

    assert results["evaluations"][0]["dataset"] == "johansen_joint"
    assert results["evaluations"][0]["source_epoch"] == 3
    assert "| `johansen_joint` | `macaque` | `lora` | 3 |" in render_markdown(results)


def test_rejects_epoch_mismatch(tmp_path):
    run = tmp_path / "zemke2023_human_lora_joint_epoch1_eval"
    run.mkdir()
    (run / "evaluation.json").write_text(
        json.dumps({"source_epoch": 2, "source_global_step": 10, "metrics": {"test": {}}})
    )

    with pytest.raises(ValueError, match="names epoch 1"):
        collate(tmp_path)
