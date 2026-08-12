from pathlib import Path

import pytest

from scripts.v0data.add_track_nonzero_means import add_track_nonzero_means


def _config() -> dict:
    return {
        "dataset": "example",
        "heads": [
            {
                "id": "atac",
                "targets": [
                    {"path": "/data/a.bw", "label": "a"},
                    {"path": "/data/b.bw", "label": "b"},
                ],
            },
            {
                "id": "rna",
                "targets": [{"path": "/data/rna.bw", "nonzero_mean": 7.0}],
            },
        ],
    }


def test_add_track_nonzero_means_only_changes_selected_heads() -> None:
    source = _config()
    means = {"a.bw": 2.0, "b.bw": 4.0}

    result = add_track_nonzero_means(
        source,
        ["atac"],
        workers=2,
        mean_fn=lambda path: means[path.name],
    )

    assert [target["nonzero_mean"] for target in result["heads"][0]["targets"]] == [
        2.0,
        4.0,
    ]
    assert result["heads"][1]["targets"][0]["nonzero_mean"] == 7.0
    assert "nonzero_mean" not in source["heads"][0]["targets"][0]
    assert result["target_contract"]["nonzero_mean_scaled_heads"] == ["atac"]


def test_add_track_nonzero_means_rejects_unknown_head() -> None:
    with pytest.raises(ValueError, match="Unknown head IDs"):
        add_track_nonzero_means(_config(), ["missing"], mean_fn=lambda _: 1.0)


@pytest.mark.parametrize("mean", [0.0, -1.0, float("inf"), float("nan")])
def test_add_track_nonzero_means_requires_positive_finite_values(mean: float) -> None:
    with pytest.raises(ValueError, match="Invalid nonzero mean"):
        add_track_nonzero_means(
            _config(),
            ["atac"],
            mean_fn=lambda _: mean,
        )
