from __future__ import annotations

from pathlib import Path

import pyBigWig
import pytest

from alphagenome_ft.finetune.target_manifest import (
    bigwig_nonzero_mean,
    build_head_config,
    retain_target_heads,
    set_gene_window_assignment,
    set_head_output_rank,
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
