import copy

import pytest

from alphagenome_ft.finetune.target_manifest import make_gene_only_config


def _config() -> dict:
    return {
        "heads": [
            {"id": "atac", "resolutions": [1, 128]},
            {
                "id": "rna",
                "resolutions": [1, 128],
                "gene_supervision": {
                    "path": "old.npz",
                    "loss_weight": 1.0,
                    "coverage_loss_weight": 0.1,
                },
            },
        ]
    }


def test_make_gene_only_config_preserves_source_and_disables_coverage() -> None:
    source = _config()
    original = copy.deepcopy(source)

    result = make_gene_only_config(source, head_id="rna", correlation_loss_weight=0.1)

    assert source == original
    rna = result["heads"][1]
    assert rna["resolutions"] == [128]
    assert rna["gene_supervision"]["coverage_loss_weight"] == 0.0
    assert rna["double_centered_correlation_loss_weight"] == 0.1
    assert rna["row_centered_correlation_loss_weight"] == 0.0


def test_make_gene_only_config_can_set_row_correlation() -> None:
    result = make_gene_only_config(
        _config(),
        head_id="rna",
        correlation_loss_weight=0.0,
        row_correlation_loss_weight=1.0,
    )

    assert result["heads"][1]["row_centered_correlation_loss_weight"] == 1.0


def test_make_gene_only_config_can_factorize_output() -> None:
    result = make_gene_only_config(
        _config(),
        head_id="rna",
        correlation_loss_weight=0.0,
        output_rank=1,
    )

    assert result["heads"][1]["output_rank"] == 1


def test_make_gene_only_config_can_share_strands_and_factorize() -> None:
    source = _config()
    source["heads"][1]["targets"] = [
        {"path": "a.plus.bw", "label": "a (+)", "strand": "+"},
        {"path": "a.minus.bw", "label": "a (-)", "strand": "-"},
        {"path": "b.plus.bw", "label": "b (+)", "strand": "+"},
        {"path": "b.minus.bw", "label": "b (-)", "strand": "-"},
    ]

    result = make_gene_only_config(
        source,
        head_id="rna",
        correlation_loss_weight=1.0,
        output_rank=1,
        unstranded_output=True,
    )

    rna = result["heads"][1]
    assert rna["output_rank"] == 1
    assert [(target["label"], target["strand"]) for target in rna["targets"]] == [
        ("a", "."),
        ("b", "."),
    ]


def test_make_gene_only_config_can_replace_supervision_path() -> None:
    result = make_gene_only_config(
        _config(),
        head_id="rna",
        correlation_loss_weight=1.0,
        gene_supervision_path="corrected.npz",
    )

    assert result["heads"][1]["gene_supervision"]["path"] == "corrected.npz"


@pytest.mark.parametrize("invalid", [-0.1, float("inf"), float("nan")])
def test_make_gene_only_config_rejects_invalid_weight(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        make_gene_only_config(_config(), head_id="rna", correlation_loss_weight=invalid)

    with pytest.raises(ValueError, match="finite and non-negative"):
        make_gene_only_config(
            _config(),
            head_id="rna",
            correlation_loss_weight=0.0,
            row_correlation_loss_weight=invalid,
        )
