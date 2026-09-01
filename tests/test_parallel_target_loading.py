from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pyBigWig

from alphagenome_ft.finetune.config import HeadSpec, TrackInfo
from alphagenome_ft.finetune.data import (
    BigWigDataModule,
    WindowedTargetCache,
    build_interval,
)
import alphagenome_ft.finetune.data as data_module


def test_target_cache_can_select_evaluation_splits_only():
    intervals = {"train": [object()], "valid": [object()], "test": [object()]}

    selected = WindowedTargetCache._select_interval_mapping(intervals, ("valid", "test"))

    assert tuple(selected) == ("valid", "test")
    assert selected["valid"] == intervals["valid"]
    assert selected["test"] == intervals["test"]


def test_target_cache_rejects_invalid_split_selection():
    intervals = {"train": [object()], "valid": [object()], "test": [object()]}

    with np.testing.assert_raises_regex(ValueError, "unknown split"):
        WindowedTargetCache._select_interval_mapping(intervals, ("valid", "holdout"))
    with np.testing.assert_raises_regex(ValueError, "duplicate split"):
        WindowedTargetCache._select_interval_mapping(intervals, ("valid", "valid"))


def test_target_cache_loads_only_selected_splits(tmp_path):
    track_path = tmp_path / "track.bw"
    with pyBigWig.open(str(track_path), "w") as track:
        track.addHeader([("chr1", 12)])
        track.addEntries(["chr1"], [0], ends=[12], values=[2.0])
    spec = HeadSpec(
        head_id="atac",
        source="predefined",
        kind="atac",
        tracks=[TrackInfo("atac", track_path)],
    )
    intervals = {
        "train": [build_interval(chromosome="chr1", start=0, end=4)],
        "valid": [build_interval(chromosome="chr1", start=4, end=8)],
        "test": [build_interval(chromosome="chr1", start=8, end=12)],
    }

    WindowedTargetCache.build(
        tmp_path / "cache",
        intervals=intervals,
        head_specs=[spec],
        workers=1,
        cache_splits=("valid", "test"),
    )
    cache = WindowedTargetCache(
        tmp_path / "cache",
        intervals=intervals,
        head_specs=[spec],
        cache_splits=("valid", "test"),
    )

    assert not cache.has_split("train")
    assert cache.has_split("valid")
    assert cache.has_split("test")
    assert not (tmp_path / "cache" / "train").exists()
    np.testing.assert_array_equal(
        cache.arrays_for_split("valid")["atac"], np.full((1, 4, 1), 2.0, dtype=np.float16)
    )


def test_target_cache_rejects_values_that_overflow_requested_dtype(tmp_path):
    track_path = tmp_path / "large.bw"
    with pyBigWig.open(str(track_path), "w") as track:
        track.addHeader([("chr1", 4)])
        track.addEntries(["chr1"], [0], ends=[4], values=[70_000.0])
    spec = HeadSpec(
        head_id="atac",
        source="predefined",
        kind="atac",
        tracks=[TrackInfo("atac", track_path)],
    )
    intervals = {"valid": [build_interval(chromosome="chr1", start=0, end=4)]}

    with np.testing.assert_raises_regex(ValueError, "Use float32 target caching"):
        WindowedTargetCache.build(
            tmp_path / "cache-f16",
            intervals=intervals,
            head_specs=[spec],
            dtype="float16",
        )

    WindowedTargetCache.build(
        tmp_path / "cache-f32",
        intervals=intervals,
        head_specs=[spec],
        dtype="float32",
    )


def test_target_cache_rejects_infinite_bigwig_values(tmp_path):
    track = TrackInfo("atac", tmp_path / "atac.bw")
    interval = build_interval(chromosome="chr1", start=0, end=4)

    class InfiniteHandle:
        def values(self, _chromosome, _start, _end, numpy=True):
            assert numpy
            return np.asarray([0.0, np.inf, 0.0, 0.0], dtype=np.float32)

    with np.testing.assert_raises_regex(ValueError, "contains an infinite value"):
        WindowedTargetCache._read_window(
            [InfiniteHandle()],
            interval,
            np.dtype("float32"),
            [track],
        )


class _FakeExtractor:
    def __init__(self, _path):
        pass

    def extract(self, _window):
        return "ACGT"


class _FakeEncoder:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def encode(self, _sequence):
        return np.eye(4, dtype=self.dtype)


class _FakeBigWig:
    def __init__(self, offset):
        self.offset = offset
        self.active = False

    def values(self, _chromosome, start, end, numpy=True):
        assert numpy
        assert not self.active, "one BigWig handle was read concurrently"
        self.active = True
        try:
            return np.arange(start, end, dtype=np.float32) + self.offset
        finally:
            self.active = False


def test_parallel_batch_reads_each_track_across_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module.fasta_lib, "FastaExtractor", _FakeExtractor)
    monkeypatch.setattr(data_module.one_hot_encoder, "DNAOneHotEncoder", _FakeEncoder)
    module = object.__new__(BigWigDataModule)
    module._fasta_path = tmp_path / "reference.fa"
    module._encoder = _FakeEncoder()
    module._head_specs = [
        HeadSpec(
            head_id="atac",
            source="predefined",
            kind="atac",
            tracks=[TrackInfo("a", tmp_path / "a.bw"), TrackInfo("b", tmp_path / "b.bw")],
        )
    ]
    module._coverage_head_specs = tuple(module._head_specs)
    module._gene_supervisions = {}
    module._max_genes = {}
    windows = [
        SimpleNamespace(chromosome="chr1", start=0, end=4),
        SimpleNamespace(chromosome="chr1", start=4, end=8),
    ]
    handles = {"atac": [_FakeBigWig(10), _FakeBigWig(20)]}

    with (
        ThreadPoolExecutor(max_workers=2) as window_executor,
        ThreadPoolExecutor(max_workers=2) as target_executor,
    ):
        batch = module._make_batch_parallel(
            [0, 1],
            windows,
            handles,
            window_executor,
            target_executor,
        )

    assert batch["sequences"].shape == (2, 4, 4)
    np.testing.assert_array_equal(
        batch["targets_atac"],
        np.asarray(
            [
                [[10, 20], [11, 21], [12, 22], [13, 23]],
                [[14, 24], [15, 25], [16, 26], [17, 27]],
            ],
            dtype=np.float32,
        ),
    )


def test_parallel_batch_omits_gene_only_coverage_targets(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module.fasta_lib, "FastaExtractor", _FakeExtractor)
    monkeypatch.setattr(data_module.one_hot_encoder, "DNAOneHotEncoder", _FakeEncoder)
    atac = HeadSpec(
        head_id="atac",
        source="predefined",
        kind="atac",
        tracks=[TrackInfo("atac", tmp_path / "atac.bw")],
    )
    rna = HeadSpec(
        head_id="rna",
        source="predefined",
        kind="rna_seq",
        tracks=[TrackInfo("rna (+)", tmp_path / "rna.bw", "+")],
        gene_supervision_path=tmp_path / "genes.npz",
        gene_loss_weight=1.0,
        coverage_loss_weight=0.0,
    )
    module = object.__new__(BigWigDataModule)
    module._fasta_path = tmp_path / "reference.fa"
    module._encoder = _FakeEncoder()
    module._head_specs = [atac, rna]
    module._coverage_head_specs = (atac,)
    module._gene_supervisions = {}
    module._max_genes = {}
    windows = [SimpleNamespace(chromosome="chr1", start=0, end=4)]

    with (
        ThreadPoolExecutor(max_workers=1) as window_executor,
        ThreadPoolExecutor(max_workers=1) as target_executor,
    ):
        batch = module._make_batch_parallel(
            [0],
            windows,
            {"atac": [_FakeBigWig(10)]},
            window_executor,
            target_executor,
        )

    assert "targets_atac" in batch
    assert "targets_rna" not in batch
