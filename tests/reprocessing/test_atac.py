import h5py
import numpy as np

from alphagenome_ft.finetune.reprocessing import (
    BinnedAtacAccumulator,
    fragment_totals_by_group,
    read_cell_groups,
)


def test_read_cell_groups_supports_categorical_groups(tmp_path):
    path = tmp_path / "cells.h5ad"
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.create_dataset("_index", data=np.asarray([b"cell-a", b"cell-b"]))
        group = obs.create_group("Group")
        group.create_dataset("categories", data=np.asarray([b"A", b"B"]))
        group.create_dataset("codes", data=np.asarray([1, 0]))

    assert read_cell_groups(path) == {"cell-a": "B", "cell-b": "A"}


def test_binned_atac_accumulator_tracks_insertions_and_coverage():
    accumulator = BinnedAtacAccumulator(
        num_groups=2,
        chromosome_size=25,
        bin_size=10,
        tn5_shift=False,
    )
    accumulator.add(
        group_indices=[0, 0, 1],
        starts=[2, 8, 20],
        ends=[7, 23, 25],
        counts=[1, 2, 1],
    )

    insertion_spmr, coverage_spmr = accumulator.normalized([3, 1])

    np.testing.assert_allclose(accumulator.insertion_counts[0], [4, 0, 2])
    np.testing.assert_allclose(accumulator.covered_bases[0], [9, 20, 6])
    np.testing.assert_allclose(accumulator.insertion_counts[1], [0, 0, 2])
    np.testing.assert_allclose(accumulator.covered_bases[1], [0, 0, 5])
    np.testing.assert_allclose(insertion_spmr[0], np.asarray([4, 0, 2]) / (3 / 1_000_000))
    np.testing.assert_allclose(coverage_spmr[0], np.asarray([0.9, 2.0, 1.2]) / (3 / 1_000_000))


def test_fragment_totals_by_group_reports_unmatched_cells():
    totals, missing = fragment_totals_by_group(
        {"cell-a": 3, "cell-b": 5, "other": 10},
        {"cell-a": "A", "cell-b": "B"},
        ["A", "B"],
    )

    np.testing.assert_array_equal(totals, [3, 5])
    assert missing == 1
