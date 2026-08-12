import gzip
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

from alphagenome_ft.finetune.reprocessing import (
    aggregate_sparse_count_chunks_by_group,
    aggregate_sparse_counts_by_group,
    aggregate_matrix_market_by_group,
    align_cpm_to_gene_supervision,
    normalize_counts_per_million,
    read_10x_features,
)


def test_aggregate_sparse_counts_by_group_and_normalize():
    counts = sparse.csr_matrix(
        np.asarray([[1, 0, 2], [0, 3, 0], [4, 0, 1]], dtype=np.float32)
    )

    groups, aggregated, n_cells = aggregate_sparse_counts_by_group(
        counts,
        ("b", "a", "b"),
    )

    assert groups == ("a", "b")
    np.testing.assert_array_equal(aggregated, [[0, 3, 0], [5, 0, 3]])
    np.testing.assert_array_equal(n_cells, [1, 2])
    np.testing.assert_allclose(
        normalize_counts_per_million(aggregated),
        [[0, 1_000_000, 0], [625_000, 0, 375_000]],
    )


def test_chunked_sparse_aggregation_matches_full_matrix():
    counts = sparse.csr_matrix(
        np.asarray(
            [[1, 0, 2], [0, 3, 0], [4, 0, 1], [2, 2, 0]],
            dtype=np.float32,
        )
    )
    labels = ("b", "a", "b", "c")

    expected = aggregate_sparse_counts_by_group(counts, labels)
    observed = aggregate_sparse_count_chunks_by_group(
        (counts[:1], counts[1:3], counts[3:]), labels
    )

    assert observed[0] == expected[0]
    np.testing.assert_array_equal(observed[1], expected[1])
    np.testing.assert_array_equal(observed[2], expected[2])


def test_chunked_sparse_aggregation_requires_all_labeled_rows():
    counts = sparse.csr_matrix(np.ones((2, 3), dtype=np.float32))

    with np.testing.assert_raises_regex(ValueError, "1 cells but 2 labels"):
        aggregate_sparse_count_chunks_by_group((counts[:1],), ("a", "b"))


def test_sparse_aggregation_promotes_before_large_count_sum():
    counts = sparse.csr_matrix(
        np.asarray([[8_388_608], [8_388_608], [1]], dtype=np.float32)
    )

    _, full, _ = aggregate_sparse_counts_by_group(counts, ("a", "a", "a"))
    _, chunked, _ = aggregate_sparse_count_chunks_by_group(
        (counts[:2], counts[2:]), ("a", "a", "a")
    )

    np.testing.assert_array_equal(full, [[16_777_217]])
    np.testing.assert_array_equal(chunked, full)


def test_align_cpm_to_gene_supervision_reorders_maps_and_renormalizes():
    aligned = align_cpm_to_gene_supervision(
        template_groups=("second", "first"),
        template_gene_ids=("native_b", "native_a"),
        source_groups=("first", "second"),
        source_gene_ids=("source_a", "source_b", "unused"),
        source_cpm=np.asarray([[10, 30, 60], [20, 20, 60]], dtype=np.float32),
        source_gene_by_template={"native_a": "source_a", "native_b": "source_b"},
    )

    np.testing.assert_allclose(aligned, [[500_000, 500_000], [750_000, 250_000]])


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
