import numpy as np
import pytest

from alphagenome_ft.finetune.reliability import (
    balanced_library_split,
    counts_per_million,
    double_centered_pearson,
    split_half_pseudobulks,
    spearman_brown,
)


def test_balanced_library_split_assigns_observed_samples_to_both_halves() -> None:
    depths = np.asarray([[10, 0], [6, 3], [4, 2], [1, 1]], dtype=float)
    assignment = balanced_library_split(depths)
    for group in range(depths.shape[1]):
        observed = depths[:, group] > 0
        assert assignment[observed, group].any()
        assert (~assignment[observed, group]).any()


def test_counts_per_million_marks_empty_groups() -> None:
    cpm, valid = counts_per_million(np.asarray([[1, 3], [0, 0]], dtype=float))
    assert valid.tolist() == [True, False]
    assert cpm[0].tolist() == [250_000.0, 750_000.0]
    assert cpm[1].tolist() == [0.0, 0.0]


def test_double_centered_pearson_and_spearman_brown() -> None:
    values = np.asarray([[1, 4, 2], [3, 0, 5], [2, 6, 1]], dtype=float)
    assert double_centered_pearson(values, values) == pytest.approx(1.0)
    assert spearman_brown(0.8) == pytest.approx(8 / 9)


def test_split_half_pseudobulks_balances_each_group_independently() -> None:
    counts = np.asarray(
        [
            [[8, 0], [0, 0]],
            [[4, 0], [0, 6]],
            [[0, 2], [0, 3]],
            [[0, 1], [2, 0]],
        ],
        dtype=float,
    )

    first, second, valid = split_half_pseudobulks(counts)

    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(first.sum(axis=1), [1_000_000, 1_000_000])
    np.testing.assert_allclose(second.sum(axis=1), [1_000_000, 1_000_000])
