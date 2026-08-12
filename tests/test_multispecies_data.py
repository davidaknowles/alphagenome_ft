import numpy as np
from alphagenome.data import genome

from alphagenome_ft.finetune.data import MultiSpeciesDataModule


class _FakeModule:
    def __init__(self, species: str, batches: int, max_genes: int):
        self.species = species
        self.batches = batches
        self._drop_last = True
        self._batch_size = 1
        self._max_genes = {"rna": max_genes}
        self._intervals = {
            "train": [genome.Interval(species, idx, idx + 1) for idx in range(batches)],
            "valid": [genome.Interval(species, 0, 1)],
        }

    def iter_batches(self, split, *, seed=None, shuffle=None):
        count = self.batches if split == "train" else 1
        for batch_idx in range(count):
            yield {
                "species": self.species,
                "batch": batch_idx,
                "seed": seed,
                "sequences": np.zeros((1, 4, 4), dtype=np.float32),
            }


def test_multispecies_batches_round_robin_and_stop_at_shortest_species():
    human = _FakeModule("human", batches=3, max_genes=5)
    macaque = _FakeModule("macaque", batches=2, max_genes=7)
    module = MultiSpeciesDataModule({"human": human, "macaque": macaque})

    batches = list(module.iter_batches("train", seed=10))

    assert [batch["species"] for batch in batches] == [
        "human",
        "macaque",
        "human",
        "macaque",
    ]
    assert human._max_genes["rna"] == macaque._max_genes["rna"] == 7
    assert module._batch_size == 1
    assert batches[0]["seed"] == 10
    assert batches[1]["seed"] == 11


def test_multispecies_batches_carry_per_species_organism_indices():
    human = _FakeModule("human", batches=1, max_genes=1)
    mouse = _FakeModule("mouse", batches=1, max_genes=1)
    module = MultiSpeciesDataModule(
        {"human": human, "mouse": mouse},
        organism_indices={"human": 0, "mouse": 1},
    )

    batches = list(module.iter_batches("train"))

    assert [int(batch["organism_index"][0]) for batch in batches] == [0, 1]


def test_single_species_wrapper_retains_organism_index_for_evaluation():
    mouse = _FakeModule("mouse", batches=1, max_genes=1)
    module = MultiSpeciesDataModule(
        {"mouse": mouse},
        organism_indices={"mouse": 1},
    )

    batches = list(module.iter_batches("valid", shuffle=False))

    assert len(batches) == 1
    assert int(batches[0]["organism_index"][0]) == 1
    assert module._intervals["valid"] == mouse._intervals["valid"]
