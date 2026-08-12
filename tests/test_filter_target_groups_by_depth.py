import pytest

from scripts.v0data.filter_target_groups_by_depth import filter_target_groups


def _config():
    return {
        "heads": [
            {
                "id": "atac",
                "targets": [
                    {"label": "a", "path": "a.bw"},
                    {"label": "b", "path": "b.bw"},
                ],
            },
            {
                "id": "rna",
                "targets": [
                    {"label": "a (+)", "path": "a.plus.bw"},
                    {"label": "a (-)", "path": "a.minus.bw"},
                    {"label": "b (+)", "path": "b.plus.bw"},
                    {"label": "b (-)", "path": "b.minus.bw"},
                ],
            },
        ]
    }


def test_filter_target_groups_keeps_modalities_aligned() -> None:
    result, retained = filter_target_groups(
        _config(),
        {"a": 9_000_000, "b": 11_000_000},
        minimum_fragments=10_000_000,
        atac_head_id="atac",
        rna_head_id="rna",
    )

    assert retained == ["b"]
    assert [target["label"] for target in result["heads"][0]["targets"]] == ["b"]
    assert [target["label"] for target in result["heads"][1]["targets"]] == [
        "b (+)",
        "b (-)",
    ]
    assert result["group_depth_filter"]["excluded_groups"] == {"a": 9_000_000}


def test_filter_target_groups_rejects_missing_paired_rna() -> None:
    config = _config()
    config["heads"][1]["targets"].pop()

    with pytest.raises(ValueError, match="missing paired labels"):
        filter_target_groups(
            config,
            {"a": 9_000_000, "b": 11_000_000},
            minimum_fragments=10_000_000,
            atac_head_id="atac",
            rna_head_id="rna",
        )
