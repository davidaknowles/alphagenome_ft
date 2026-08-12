import copy

import pytest

from scripts.v0data.prepare_gene_only_rna_config import make_gene_only_config


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


def test_make_gene_only_config_can_replace_supervision_path() -> None:
    result = make_gene_only_config(
        _config(),
        head_id="rna",
        correlation_loss_weight=1.0,
        gene_supervision_path="corrected.npz",
    )

    assert result["heads"][1]["gene_supervision"]["path"] == "corrected.npz"


def test_make_gene_only_config_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_gene_only_config(_config(), head_id="rna", correlation_loss_weight=-0.1)
