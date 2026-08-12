from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.io import mmwrite

from scripts.v0data.zemke2023_rna_reprocessing.audit_donor_reliability import (
    audit_reliability,
    read_donor_group_labels,
)


def test_read_donor_group_labels_filters_and_normalizes_groups(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text(
        "cell\torig.ident\tsubclass\n"
        "cell1\tdonor2\tL2/3 IT\n"
        "cell2\tdonor1\tODC\n"
        "cell3\tdonor1\tunsupported\n"
    )
    labels, donors = read_donor_group_labels(
        metadata, valid_groups=("L2_3_IT", "ODC")
    )
    assert labels == {
        "cell1": "donor2\x1fL2_3_IT",
        "cell2": "donor1\x1fODC",
    }
    assert donors == ("donor1", "donor2")


def test_read_donor_group_labels_rejects_duplicate_cells(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text(
        "cell\torig.ident\tsubclass\n"
        "cell1\tdonor1\tODC\n"
        "cell1\tdonor2\tODC\n"
    )
    with pytest.raises(ValueError, match="Duplicate metadata barcode"):
        read_donor_group_labels(metadata, valid_groups=("ODC",))


def test_audit_reliability_uses_donor_group_and_modeled_gene_axes(
    tmp_path: Path,
) -> None:
    barcodes = tuple(f"cell{index}" for index in range(8))
    donors = ("d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4")
    groups = ("A", "B") * 4
    matrix = np.asarray(
        [
            [8, 2, 7, 3, 6, 4, 5, 5],
            [1, 9, 2, 8, 3, 7, 4, 6],
            [4, 1, 5, 2, 6, 3, 7, 4],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ],
        dtype=np.int64,
    )
    matrix_path = tmp_path / "matrix.mtx"
    mmwrite(matrix_path, sparse.coo_matrix(matrix))
    barcode_path = tmp_path / "barcodes.tsv"
    barcode_path.write_text("\n".join(barcodes) + "\n")
    feature_path = tmp_path / "features.tsv"
    feature_path.write_text("g1\ng2\ng3\nunused\n")
    metadata_path = tmp_path / "metadata.tsv"
    metadata_path.write_text(
        "cell\torig.ident\tsubclass\n"
        + "".join(
            f"{cell}\t{donor}\t{group}\n"
            for cell, donor, group in zip(barcodes, donors, groups, strict=True)
        )
    )
    supervision_path = tmp_path / "supervision.npz"
    np.savez_compressed(
        supervision_path,
        groups=np.asarray(("A", "B", "unsupported")),
        group_valid=np.asarray((True, True, False)),
        gene_ids=np.asarray(("g1", "g2", "g3")),
    )

    result = audit_reliability(
        matrix_path=matrix_path,
        barcode_path=barcode_path,
        feature_path=feature_path,
        metadata_path=metadata_path,
        supervision_path=supervision_path,
        species="test",
    )
    assert result["donors"] == 4
    assert result["groups"] == 2
    assert result["genes"] == 3
    assert result["groups_estimable_in_both_halves"] == 2
    assert np.isfinite(result["raw_cpm_double_centered_r"])
