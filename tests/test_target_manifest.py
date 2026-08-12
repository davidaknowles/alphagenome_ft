from __future__ import annotations

from pathlib import Path

import pyBigWig
import pytest

from alphagenome_ft.finetune.target_manifest import (
    bigwig_nonzero_mean,
    build_head_config,
    combined_bigwig_nonzero_mean,
    make_gene_only_config,
    retain_target_heads,
    set_gene_window_assignment,
    set_head_output_rank,
)


def test_make_gene_only_config_can_collapse_strand_pairs(tmp_path: Path) -> None:
    group_a_plus = tmp_path / "group_a.plus.bw"
    group_a_minus = tmp_path / "group_a.minus.bw"
    _write_bigwig(group_a_plus)
    _write_bigwig(group_a_minus)
    source = {
        "heads": [
            {
                "id": "rna",
                "kind": "rna_seq",
                "resolutions": [1, 128],
                "gene_supervision": {"path": "genes.npz", "coverage_loss_weight": 0.1},
                "targets": [
                    {
                        "path": str(group_a_plus),
                        "label": "group_a (+)",
                        "strand": "+",
                        "nonzero_mean": 2.0,
                    },
                    {
                        "path": str(group_a_minus),
                        "label": "group_a (-)",
                        "strand": "-",
                        "nonzero_mean": 4.0,
                    },
                    {"path": "group_b.plus.bw", "label": "group_b (+)", "strand": "+"},
                    {"path": "group_b.minus.bw", "label": "group_b (-)", "strand": "-"},
                ],
            }
        ]
    }

    result = make_gene_only_config(
        source,
        head_id="rna",
        correlation_loss_weight=0.0,
        unstranded_output=True,
    )

    head = result["heads"][0]
    assert head["gene_supervision"]["coverage_loss_weight"] == 0.0
    assert head["resolutions"] == [128]
    assert [(target["label"], target["strand"]) for target in head["targets"]] == [
        ("group_a", "."),
        ("group_b", "."),
    ]
    assert head["targets"][0]["nonzero_mean"] == pytest.approx(6.5)
    assert source["heads"][0]["targets"][0]["strand"] == "+"


def test_make_gene_only_config_rejects_nonpaired_unstranded_source() -> None:
    source = {
        "heads": [
            {
                "id": "rna",
                "gene_supervision": {"path": "genes.npz"},
                "targets": [{"path": "a.bw", "strand": "."}],
            }
        ]
    }

    with pytest.raises(ValueError, match="interleaved"):
        make_gene_only_config(
            source,
            head_id="rna",
            correlation_loss_weight=0.0,
            unstranded_output=True,
        )


def test_retain_target_heads_preserves_source_order_without_mutation() -> None:
    source = {"dataset": "study", "heads": [{"id": "atac"}, {"id": "rna"}]}

    result = retain_target_heads(source, ("rna",))

    assert result["heads"] == [{"id": "rna"}]
    assert source["heads"] == [{"id": "atac"}, {"id": "rna"}]


def test_set_head_output_rank_changes_only_requested_rna_head() -> None:
    source = {
        "heads": [
            {"id": "atac", "kind": "atac", "targets": [{}, {}]},
            {"id": "rna", "kind": "rna_seq", "targets": [{}, {}, {}, {}]},
        ]
    }

    result = set_head_output_rank(source, head_id="rna", output_rank=2)

    assert result["heads"][1]["output_rank"] == 2
    assert "output_rank" not in source["heads"][1]


def test_set_gene_window_assignment_changes_only_requested_gene_head() -> None:
    source = {
        "heads": [
            {"id": "atac"},
            {"id": "rna", "gene_supervision": {"path": "genes.npz"}},
        ]
    }

    result = set_gene_window_assignment(
        source,
        head_id="rna",
        assignment="max_exon_overlap_scaled",
    )

    assert result["heads"][1]["gene_supervision"]["window_assignment"] == (
        "max_exon_overlap_scaled"
    )
    assert "window_assignment" not in source["heads"][1]["gene_supervision"]


def _write_bigwig(path: Path) -> None:
    bigwig = pyBigWig.open(str(path), "w")
    bigwig.addHeader([("chr1", 10)])
    bigwig.addEntries(
        ["chr1", "chr1", "chr1"],
        [0, 2, 5],
        ends=[2, 5, 10],
        values=[0.0, 2.0, 4.0],
    )
    bigwig.close()


def test_combined_bigwig_nonzero_mean_sums_overlapping_tracks(tmp_path: Path) -> None:
    first = tmp_path / "first.bw"
    second = tmp_path / "second.bw"
    _write_bigwig(first)
    _write_bigwig(second)

    assert combined_bigwig_nonzero_mean((first, second)) == pytest.approx(6.5)


def test_bigwig_nonzero_mean_is_base_weighted(tmp_path: Path) -> None:
    path = tmp_path / "target.bw"
    _write_bigwig(path)
    assert bigwig_nonzero_mean(path) == pytest.approx((3 * 2.0 + 5 * 4.0) / 8)


def test_build_rna_head_includes_scaling(tmp_path: Path) -> None:
    path = tmp_path / "target.bw"
    _write_bigwig(path)
    head = build_head_config(
        head_id="rna",
        kind="rna_seq",
        tracks=[path],
        labels=["cell"],
        nonzero_means=[3.25],
    )
    assert head["apply_squashing"] is True
    assert head["targets"][0]["nonzero_mean"] == 3.25
