import json
from pathlib import Path

import pytest

from scripts.v0data.assert_gradient_diagnostics import assert_head_gradients


def _write(path: Path, norms: dict[str, float]) -> Path:
    path.write_text(
        json.dumps(
            {
                "heads": {
                    head: {"head_gradient_norm": norm}
                    for head, norm in norms.items()
                }
            }
        )
    )
    return path


def test_accepts_positive_expected_head_gradients(tmp_path: Path) -> None:
    path = _write(tmp_path / "gradients.json", {"atac": 0.2, "rna": 0.3})

    result = assert_head_gradients(
        [path],
        expected_heads={"atac", "rna"},
        minimum_norm=1e-8,
    )

    assert result[str(path)] == {"atac": 0.2, "rna": 0.3}


@pytest.mark.parametrize("norm", [0.0, float("nan"), float("inf")])
def test_rejects_disconnected_head_gradient(tmp_path: Path, norm: float) -> None:
    path = _write(tmp_path / "gradients.json", {"atac": 0.2, "rna": norm})

    with pytest.raises(ValueError, match="gate failed"):
        assert_head_gradients(
            [path],
            expected_heads={"atac", "rna"},
            minimum_norm=0.0,
        )


def test_rejects_missing_expected_head(tmp_path: Path) -> None:
    path = _write(tmp_path / "gradients.json", {"atac": 0.2})

    with pytest.raises(ValueError, match="Missing heads"):
        assert_head_gradients(
            [path],
            expected_heads={"atac", "rna"},
            minimum_norm=0.0,
        )
