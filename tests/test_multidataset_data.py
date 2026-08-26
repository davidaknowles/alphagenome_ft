import numpy as np

from alphagenome_ft.finetune.data import (
    BigWigDataModule,
    MultiDatasetDataModule,
    build_interval,
    exclude_overlapping_intervals,
    load_excluded_regions_from_bed,
)


class _Spec:
    def __init__(self, head_id):
        self.head_id = head_id


class _Module:
    def __init__(self, name, head_names, batch_count, *, batch_size=2):
        self.name = name
        self._head_specs = [_Spec(head_name) for head_name in head_names]
        self._batch_count = batch_count
        self._batch_size = batch_size
        self._drop_last = True
        self._max_genes = {head_name: batch_count for head_name in head_names}
        self._intervals = {
            split: [f"{name}-{split}-{index}" for index in range(batch_count * batch_size)]
            for split in ("train", "valid")
        }

    def num_batches_per_epoch(self, split):
        return self._batch_count

    def iter_batches(self, split, *, seed=None, shuffle=None):
        for index in range(self._batch_count):
            yield {
                "sequences": np.full((self._batch_size, 1, 4), index, dtype=np.float32),
                "negative_strand_mask": np.zeros((self._batch_size,), dtype=bool),
                "batch_index": np.asarray(index),
            }


def test_single_dataset_batch_count_rounds_partial_batch():
    module = BigWigDataModule.__new__(BigWigDataModule)
    module._intervals = {"valid": list(range(9))}
    module._batch_size = 4
    module._drop_last = False

    assert module.num_batches_per_epoch("valid") == 3

    module._drop_last = True
    assert module.num_batches_per_epoch("valid") == 2


def test_excluded_regions_remove_only_overlapping_windows(tmp_path):
    bed = tmp_path / "excluded.bed"
    bed.write_text("chr9\t150\t250\tANO6\n")
    intervals = {
        "valid": [build_interval(chromosome="chr8", start=100, end=200)],
        "test": [
            build_interval(chromosome="chr9", start=0, end=100),
            build_interval(chromosome="chr9", start=100, end=200),
            build_interval(chromosome="chr9", start=250, end=350),
        ],
    }

    filtered = exclude_overlapping_intervals(
        intervals,
        load_excluded_regions_from_bed(bed),
    )

    assert filtered["valid"] == intervals["valid"]
    assert filtered["test"] == [intervals["test"][0], intervals["test"][2]]


def test_multidataset_training_balances_datasets_and_native_sources():
    hda = _Module("hda", ("hda_atac", "hda_rna"), 2)
    zemke_human = _Module("zemke_human", ("zemke_atac", "zemke_rna"), 3)
    zemke_mouse = _Module("zemke_mouse", ("zemke_atac", "zemke_rna"), 1)
    module = MultiDatasetDataModule(
        {
            "hda": {"hda_human": hda},
            "zemke": {
                "zemke_human": zemke_human,
                "zemke_mouse": zemke_mouse,
            },
        },
        organism_indices={"hda_human": 0, "zemke_human": 0, "zemke_mouse": 1},
    )

    batches = list(module.iter_batches("train", seed=7))

    assert module.num_batches_per_epoch("train") == 6
    assert len(batches) == 6
    assert [batch["_dataset_name"] for batch in batches].count("hda") == 3
    assert [batch["_dataset_name"] for batch in batches].count("zemke") == 3
    assert [batch["_source_name"] for batch in batches if batch["_dataset_name"] == "zemke"] == [
        "zemke_human",
        "zemke_mouse",
        "zemke_human",
    ]
    assert batches[0]["_active_head_names"] == ("hda_atac", "hda_rna")
    assert batches[1]["_active_head_names"] == ("zemke_atac", "zemke_rna")
    assert np.all(batches[3]["organism_index"] == 1)


def test_multidataset_training_can_balance_native_sources():
    hda = _Module("hda", ("hda_atac", "hda_rna"), 2)
    zemke_human = _Module("zemke_human", ("zemke_atac", "zemke_rna"), 3)
    zemke_mouse = _Module("zemke_mouse", ("zemke_atac", "zemke_rna"), 1)
    module = MultiDatasetDataModule(
        {
            "hda": {"hda_human": hda},
            "zemke": {
                "zemke_human": zemke_human,
                "zemke_mouse": zemke_mouse,
            },
        },
        organism_indices={"hda_human": 0, "zemke_human": 0, "zemke_mouse": 1},
        sampling_strategy="equal_sources",
    )

    batches = list(module.iter_batches("train", seed=7))

    assert module.num_batches_per_epoch("train") == 9
    assert len(batches) == 9
    assert [batch["_source_name"] for batch in batches] == [
        "hda_human",
        "zemke_human",
        "zemke_mouse",
    ] * 3


def test_multidataset_rejects_unknown_sampling_strategy():
    with np.testing.assert_raises_regex(ValueError, "sampling_strategy"):
        MultiDatasetDataModule(
            {"hda": {"hda_human": _Module("hda", ("hda_atac",), 2)}},
            sampling_strategy="largest_dataset",
        )


def test_multidataset_evaluation_visits_each_batch_once():
    first = _Module("first", ("first_head",), 2)
    second = _Module("second", ("second_head",), 1)
    module = MultiDatasetDataModule(
        {"first": {"first_source": first}, "second": {"second_source": second}}
    )

    batches = list(module.iter_batches("valid", shuffle=False))

    assert module.num_batches_per_epoch("valid") == 3
    assert [batch["_source_name"] for batch in batches] == [
        "first_source",
        "first_source",
        "second_source",
    ]


def test_multidataset_aligns_gene_padding_only_for_shared_heads():
    first = _Module("first", ("shared", "first_only"), 2)
    second = _Module("second", ("shared", "second_only"), 5)

    MultiDatasetDataModule(
        {"dataset": {"first": first, "second": second}}
    )

    assert first._max_genes == {"shared": 5, "first_only": 2}
    assert second._max_genes == {"shared": 5, "second_only": 5}
