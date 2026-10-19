import numpy as np
import pandas as pd

from alphagenome_ft.finetune.fold_split import (
    _overlap_bp,
    label_windows_direct,
    label_windows_lifted,
    split_overlap,
    tile_windows,
)


def test_overlap_bp_matches_brute_force():
    rng = np.random.default_rng(0)
    region_starts = np.array([10, 50, 200])
    region_ends = np.array([30, 120, 260])
    covered = np.zeros(400, dtype=bool)
    for start, end in zip(region_starts, region_ends):
        covered[start:end] = True
    starts = rng.integers(0, 300, size=200)
    ends = starts + rng.integers(1, 100, size=200)
    expected = np.array([covered[s:e].sum() for s, e in zip(starts, ends)])
    np.testing.assert_array_equal(_overlap_bp(starts, ends, region_starts, region_ends), expected)


def test_split_overlap_merges_overlapping_regions():
    regions = pd.DataFrame({"chrom": "chr1", "start": [0, 50], "end": [100, 150], "split": "test"})
    windows = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [200]})
    assert split_overlap(windows, regions).loc[0, "test"] == 150


def test_direct_labels_drop_windows_that_straddle_splits():
    regions = pd.DataFrame({
        "chrom": "chr1",
        "start": [0, 1000, 2000],
        "end": [1000, 2000, 3000],
        "split": ["train", "test", "valid"],
    })
    windows = tile_windows({"chr1": 3000, "chr1_random": 5000}, window_size=500, stride=500)
    labels = label_windows_direct(windows, regions).tolist()
    assert labels == ["train", "train", "test", "test", "valid", "valid"]
    straddle = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [800, 1800], "end": [1300, 2300]})
    assert label_windows_direct(straddle, regions).isna().all()


def test_lifted_labels_tolerate_small_contamination():
    regions = pd.DataFrame({"chrom": "chr1", "start": [0, 10_000], "end": [10_000, 20_000], "split": ["train", "test"]})
    windows = pd.DataFrame({"chrom": ["chrA", "chrA", "chrA"], "start": [0, 100_000, 200_000], "end": [10_000, 110_000, 210_000]})
    lifted = pd.DataFrame({
        "window": [0] * 10 + [1] * 10 + [2] * 3,
        "chrom": "chr1",
        "start": list(range(10_000, 20_000, 1000)) + list(range(10_000, 19_000, 1000)) + [500] + [3000, 4000, 5000],
    })
    lifted["end"] = lifted.start + 1000
    labels = label_windows_lifted(windows, lifted, regions, max_other=0.05).tolist()
    # Window 1 has 1 of 10 bins in train (10% > 5%), so it is dropped.
    assert labels == ["test", None, "train"]
