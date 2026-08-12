import pytest

from scripts.v0data.prepare_head_objective_config import set_head_objective


def _config() -> dict:
    return {"heads": [{"id": "atac"}, {"id": "rna", "loss_weight": 1.0}]}


def test_sets_only_requested_head_objective_weights() -> None:
    source = _config()

    result = set_head_objective(
        source,
        "rna",
        loss_weight=2.0,
        correlation_loss_weight=10.0,
    )

    assert result["heads"][0] == {"id": "atac"}
    assert result["heads"][1]["loss_weight"] == 2.0
    assert result["heads"][1]["double_centered_correlation_loss_weight"] == 10.0
    assert "double_centered_correlation_loss_weight" not in source["heads"][1]


def test_requires_an_objective_change() -> None:
    with pytest.raises(ValueError, match="At least one"):
        set_head_objective(_config(), "rna")


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_rejects_invalid_correlation_weight(value: float) -> None:
    with pytest.raises(ValueError, match="Correlation loss weight"):
        set_head_objective(_config(), "rna", correlation_loss_weight=value)


def test_rejects_ambiguous_head() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        set_head_objective({"heads": [{"id": "rna"}, {"id": "rna"}]}, "rna", loss_weight=1)
