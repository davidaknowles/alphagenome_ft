from __future__ import annotations

from pathlib import Path

import pyBigWig
import pytest

from alphagenome_ft.finetune.target_manifest import bigwig_nonzero_mean, build_head_config


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
