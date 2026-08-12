import numpy as np
import pytest

from scripts.v0data.zemke2023_rna_reprocessing.audit_gene_track_agreement import (
    double_centered_r,
    row_pattern_correlations,
)


def test_double_centered_r_is_invariant_to_row_and_column_offsets() -> None:
    values = np.asarray(
        [[1.0, 3.0, 2.0], [4.0, 2.0, 8.0], [3.0, 7.0, 5.0]]
    )
    row_offsets = np.asarray([[10.0], [20.0], [30.0]])
    column_offsets = np.asarray([[100.0, 200.0, 300.0]])

    assert double_centered_r(values, 2 * values + row_offsets + column_offsets) == pytest.approx(1.0)


def test_double_centered_r_rejects_zero_variance() -> None:
    with pytest.raises(ValueError, match="nonzero variance"):
        double_centered_r(np.ones((2, 2)), np.ones((2, 2)))


def test_row_pattern_correlations_remove_gene_specific_scale_and_offset() -> None:
    values = np.asarray([[1.0, 3.0, 2.0], [8.0, 2.0, 5.0]])
    transformed = values * np.asarray([[5.0], [0.25]]) + np.asarray([[20.0], [-3.0]])

    np.testing.assert_allclose(row_pattern_correlations(values, transformed), 1.0)
