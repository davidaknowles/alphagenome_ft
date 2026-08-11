import gzip
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

from alphagenome_ft.finetune.reprocessing import (
    aggregate_matrix_market_by_group,
    read_10x_features,
)


def test_aggregate_matrix_market_by_group(tmp_path: Path):
    matrix = sparse.coo_matrix(
        np.asarray(
            [
                [1, 2, 4, 8],
                [0, 3, 0, 5],
                [7, 0, 6, 0],
            ],
            dtype=np.float32,
        )
    )
    matrix_path = tmp_path / "matrix.mtx"
    mmwrite(matrix_path, matrix)
    barcode_path = tmp_path / "barcodes.tsv.gz"
    with gzip.open(barcode_path, "wt") as handle:
        handle.write("a\nb\nc\nd\n")

    values = aggregate_matrix_market_by_group(
        matrix_path,
        barcode_path,
        {"a": "x", "c": "x", "d": "y"},
        ("x", "y"),
    )

    np.testing.assert_allclose(values, [[5, 0, 13], [8, 5, 0]])


def test_read_10x_features_uses_first_column(tmp_path: Path):
    path = tmp_path / "features.tsv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("ENSG1.2\tGENE1\tGene Expression\nENSG2\tGENE2\tGene Expression\n")
    assert read_10x_features(path) == ("ENSG1.2", "ENSG2")
