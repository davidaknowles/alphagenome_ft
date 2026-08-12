import numpy as np
import pytest

from alphagenome_ft.finetune.reliability import (
    balanced_library_split,
    binomial_count_split,
    counts_per_million,
    double_centered_pearson,
    double_centered_rank_summary,
    split_half_pseudobulks,
    spearman_brown,
)


def test_binomial_count_split_is_deterministic_and_conservative() -> None:
    counts = np.asarray([[0, 1, 5], [12, 3, 2]], dtype=np.int64)
    first, second = binomial_count_split(counts, seed=17)
    repeated_first, repeated_second = binomial_count_split(counts, seed=17)

    np.testing.assert_array_equal(first + second, counts)
    np.testing.assert_array_equal(first, repeated_first)
    np.testing.assert_array_equal(second, repeated_second)


def test_binomial_count_split_rejects_fractional_counts() -> None:
    with pytest.raises(ValueError, match="integer-valued"):
        binomial_count_split(np.asarray([[1.5, 2.0]]), seed=0)


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


def test_double_centered_rank_summary_recovers_rank_one_target() -> None:
    observations = np.asarray([-2.0, -1.0, 1.0, 2.0])[:, None]
    tracks = np.asarray([-1.0, 0.0, 1.0])[None, :]
    values = observations @ tracks + np.arange(4.0)[:, None] + np.asarray([4.0, 8.0, 2.0])

    summary = double_centered_rank_summary(values, ranks=(1, 2))

    assert summary["numerical_rank"] == 1
    assert summary["rank_correlation_ceiling"]["1"] == pytest.approx(1.0)
    assert summary["rank_for_correlation"]["0.95"] == 1


def test_double_centered_rank_summary_rejects_constant_target() -> None:
    with pytest.raises(ValueError, match="no variance"):
        double_centered_rank_summary(np.ones((3, 4)))


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
