from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from alphagenome_ft.finetune.config import HeadSpec, TrackInfo
from alphagenome_ft.finetune.data import BigWigDataModule
import alphagenome_ft.finetune.data as data_module


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
