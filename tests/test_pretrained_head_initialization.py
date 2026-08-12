from types import SimpleNamespace

import jax.numpy as jnp
import pandas as pd

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model

from alphagenome_ft import custom_model


def test_bootstrap_track_indices_pair_stranded_channels() -> None:
    source_strands = ("+", "+", "-", "-", ".")
    target_strands = ("+", "-", "+", "-", ".")

    indices = custom_model._bootstrap_track_indices(
        source_strands,
        target_strands,
        source_valid=(True, True, True, True, False),
        seed=17,
    )

    assert indices[1] - indices[0] == 2
    assert indices[3] - indices[2] == 2
    assert indices[4] in {0, 1, 2, 3}


def test_pretrained_bootstrap_copies_consistent_output_channels(monkeypatch) -> None:
    output_type = dna_output.OutputType.RNA_SEQ
    source_strands = ("+", "+", "-", "-", ".", ".")
    target_strands = ("+", "-", "+", "-")

    class Metadata:
        def __init__(self) -> None:
            self.frame = pd.DataFrame({"strand": source_strands})
            self.padding = {output_type: jnp.asarray((False,) * len(source_strands))}

        def get(self, requested):
            return self.frame if requested == output_type else None

    organism = dna_model.Organism.HOMO_SAPIENS
    metadata = {organism: Metadata()}
    target_frame = pd.DataFrame({"strand": target_strands})
    monkeypatch.setattr(
        custom_model,
        "_resolve_user_metadata",
        lambda **_kwargs: {organism: target_frame},
    )
    source = jnp.arange(1 * 2 * 6, dtype=jnp.float32).reshape((1, 2, 6))
    params = {
        "alphagenome/head/rna_seq/resolution_128/multi_organism_linear": {
            "w": source,
        },
        "head/study_rna/resolution_128/multi_organism_linear": {
            "w": jnp.zeros((1, 2, 4), dtype=jnp.float32),
        },
    }

    result = custom_model._initialize_heads_from_pretrained_bootstrap(
        params,
        head_names=("study_rna",),
        head_configs={"study_rna": SimpleNamespace(output_type=output_type)},
        pretrained_metadata=metadata,
    )

    seed = custom_model.zlib.crc32(f"study_rna:{organism.name}".encode("utf-8"))
    indices = custom_model._bootstrap_track_indices(
        source_strands,
        target_strands,
        source_valid=(True,) * 6,
        seed=seed,
    )
    expected = jnp.take(source[0], jnp.asarray(indices), axis=-1)[None]
    actual = result["head/study_rna/resolution_128/multi_organism_linear"]["w"]
    assert jnp.array_equal(actual, expected)


def test_pretrained_bootstrap_uses_mouse_source_for_mouse_only_head(monkeypatch) -> None:
    output_type = dna_output.OutputType.ATAC
    strands = (".", ".")

    class Metadata:
        def __init__(self) -> None:
            self.frame = pd.DataFrame({"strand": strands})
            self.padding = {output_type: jnp.asarray((False, False))}

        def get(self, requested):
            return self.frame if requested == output_type else None

    human = dna_model.Organism.HOMO_SAPIENS
    mouse = dna_model.Organism.MUS_MUSCULUS
    metadata = {human: Metadata(), mouse: Metadata()}
    monkeypatch.setattr(
        custom_model,
        "_resolve_user_metadata",
        lambda **_kwargs: {mouse: pd.DataFrame({"strand": (".",)})},
    )
    params = {
        "alphagenome/head/atac/resolution_128/multi_organism_linear": {
            "w": jnp.asarray([[[1.0, 2.0]], [[101.0, 102.0]]]),
        },
        "head/mouse_atac/resolution_128/multi_organism_linear": {
            "w": jnp.zeros((1, 1, 1), dtype=jnp.float32),
        },
    }

    result = custom_model._initialize_heads_from_pretrained_bootstrap(
        params,
        head_names=("mouse_atac",),
        head_configs={"mouse_atac": SimpleNamespace(output_type=output_type)},
        pretrained_metadata=metadata,
    )

    actual = result["head/mouse_atac/resolution_128/multi_organism_linear"]["w"]
    assert actual.item() in {101.0, 102.0}
