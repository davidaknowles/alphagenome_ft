import json

import pytest

from scripts.v0data.collate_gradient_diagnostics import collate, render_markdown


def test_collates_gradient_diagnostics(tmp_path) -> None:
    run = tmp_path / "study_lora_gradnorm"
    run.mkdir()
    (run / "gradient_diagnostics.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "global_step_before_update": 0,
                "heads": {
                    "atac": {
                        "loss": 2.0,
                        "adapter_gradient_norm": 3.0,
                        "weighted_adapter_gradient_norm": 6.0,
                        "head_gradient_norm": 4.0,
                    }
                },
                "adapter_gradient_cosines": {"atac__rna": -0.25},
            }
        )
    )

    result = collate(tmp_path)
    markdown = render_markdown(result)

    assert result["runs"][0]["global_step_before_update"] == 0
    assert "| `study_lora_gradnorm` | `atac` | 2 | 3 | 6 | 4 |" in markdown
    assert "| `study_lora_gradnorm` | `atac__rna` | -0.25 |" in markdown


def test_rejects_missing_heads(tmp_path) -> None:
    run = tmp_path / "invalid"
    run.mkdir()
    (run / "gradient_diagnostics.json").write_text(json.dumps({"heads": {}}))

    with pytest.raises(ValueError, match="per-head diagnostics"):
        collate(tmp_path)
