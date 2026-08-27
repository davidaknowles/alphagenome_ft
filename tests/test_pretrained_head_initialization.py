from types import SimpleNamespace

import jax.numpy as jnp
import pandas as pd

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib

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


def test_neural_source_valid_filters_metadata_and_renal_cortex() -> None:
    source = pd.DataFrame(
        {
            "strand": ("+", "+", "+", "-", "-", "-", "."),
            "biosample_name": (
                "brain",
                "astrocyte",
                "kidney cortex",
                "brain",
                "glutamatergic neuron",
                "renal cortex interstitium",
                "liver",
            ),
        }
    )

    actual = custom_model._neural_source_valid(
        source,
        source_valid=(True,) * len(source),
        target_strands=("+", "-"),
    )

    assert actual == (True, True, False, True, True, False, False)


def test_neural_source_candidate_mask_does_not_apply_diversity_fallback() -> None:
    source = pd.DataFrame(
        {
            "strand": (".", ".", "."),
            "biosample_name": ("motor neuron", "liver", "heart"),
        }
    )

    actual = custom_model._neural_source_candidate_mask(
        source,
        source_valid=(True, True, False),
    )

    assert actual == (True, False, False)


def test_neural_source_valid_falls_back_when_pool_lacks_diversity() -> None:
    source = pd.DataFrame(
        {
            "strand": (".", ".", "."),
            "biosample_name": ("motor neuron", "liver", "heart"),
        }
    )
    valid = (True, True, False)

    actual = custom_model._neural_source_valid(
        source,
        source_valid=valid,
        target_strands=(".",),
    )

    assert actual == valid


def test_neural_target_channels_are_target_aware() -> None:
    targets = pd.DataFrame(
        {
            "name": (
                "BR_4 (+)",
                "LI_4 (+)",
                "Astrocyte (+)",
                "Microglia (+)",
                "Endo (+)",
                "PVALB (+)",
                "L5_6_NP",
                "Chandelier_all",
                "DG_all",
                "NR2F2_all",
            )
        }
    )

    actual = custom_model._neural_target_channels(targets)

    assert actual == (True, False, True, False, False, True, True, True, True, True)


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


def test_neural_bootstrap_copies_only_neural_source_channels(monkeypatch) -> None:
    output_type = dna_output.OutputType.RNA_SEQ
    source_frame = pd.DataFrame(
        {
            "strand": ("+", "+", "+", "-", "-", "-"),
            "biosample_name": (
                "brain",
                "astrocyte",
                "liver",
                "brain",
                "glutamatergic neuron",
                "heart",
            ),
        }
    )

    class Metadata:
        def __init__(self) -> None:
            self.padding = {output_type: jnp.asarray((False,) * len(source_frame))}

        def get(self, requested):
            return source_frame if requested == output_type else None

    organism = dna_model.Organism.HOMO_SAPIENS
    monkeypatch.setattr(
        custom_model,
        "_resolve_user_metadata",
        lambda **_kwargs: {
            organism: pd.DataFrame(
                {
                    "name": ("Astrocyte (+)", "GABA (-)"),
                    "strand": ("+", "-"),
                }
            )
        },
    )
    source = jnp.arange(6, dtype=jnp.float32).reshape((1, 1, 6))
    params = {
        "alphagenome/head/rna_seq/resolution_128/multi_organism_linear": {"w": source},
        "head/study_rna/resolution_128/multi_organism_linear": {
            "w": jnp.zeros((1, 1, 2), dtype=jnp.float32),
        },
    }

    result = custom_model._initialize_heads_from_pretrained_bootstrap(
        params,
        head_names=("study_rna",),
        head_configs={"study_rna": SimpleNamespace(output_type=output_type)},
        pretrained_metadata={organism: Metadata()},
        neural_sources=True,
    )

    actual = set(
        result["head/study_rna/resolution_128/multi_organism_linear"]["w"]
        .reshape(-1)
        .tolist()
    )
    assert actual <= {0.0, 1.0, 3.0, 4.0}
    assert len(actual) == 2


def test_neural_accessibility_bootstrap_copies_dnase_into_atac(monkeypatch) -> None:
    target_output_type = dna_output.OutputType.ATAC
    source_output_type = dna_output.OutputType.DNASE
    source_frame = pd.DataFrame(
        {
            "strand": (".", ".", "."),
            "biosample_name": ("brain", "motor neuron", "liver"),
        }
    )

    class Metadata:
        def __init__(self) -> None:
            self.padding = {source_output_type: jnp.asarray((False, False, False))}

        def get(self, requested):
            return source_frame if requested == source_output_type else None

    organism = dna_model.Organism.HOMO_SAPIENS
    monkeypatch.setattr(
        custom_model,
        "_resolve_user_metadata",
        lambda **_kwargs: {
            organism: pd.DataFrame({"name": ("Astrocyte", "PVALB"), "strand": (".", ".")})
        },
    )
    source = jnp.arange(3, dtype=jnp.float32).reshape((1, 1, 3))
    params = {
        "alphagenome/head/dnase/resolution_128/multi_organism_linear": {"w": source},
        "head/study_atac/resolution_128/multi_organism_linear": {
            "w": jnp.zeros((1, 1, 2), dtype=jnp.float32),
        },
    }

    result = custom_model._initialize_heads_from_pretrained_bootstrap(
        params,
        head_names=("study_atac",),
        head_configs={"study_atac": SimpleNamespace(output_type=target_output_type)},
        pretrained_metadata={organism: Metadata()},
        neural_sources=True,
        dnase_for_atac=True,
    )

    actual = set(
        result["head/study_atac/resolution_128/multi_organism_linear"]["w"]
        .reshape(-1)
        .tolist()
    )
    assert actual == {0.0, 1.0}


def test_semantic_bootstrap_prefers_cerebellum_for_purkinje() -> None:
    source = pd.DataFrame(
        {
            "name": ("DNase-seq", "DNase-seq", "DNase-seq"),
            "biosample_name": ("brain microvascular endothelial cell", "cerebellum", "liver"),
            "strand": (".", ".", "."),
        }
    )
    target = pd.DataFrame({"name": ("Neur_Purk_2",), "strand": (".",)})

    actual = custom_model._semantic_bootstrap_track_indices(
        source,
        target,
        source_valid=(True, True, True),
        source_valid_by_target=((True, True, False),),
        seed=3,
    )

    assert actual == (1,)


def test_semantic_bootstrap_preserves_biosample_across_strands() -> None:
    source = pd.DataFrame(
        {
            "name": (
                "cerebellum RNA",
                "frontal cortex RNA",
                "cerebellum RNA",
                "frontal cortex RNA",
            ),
            "biosample_name": ("cerebellum", "frontal cortex", "cerebellum", "frontal cortex"),
            "strand": ("+", "+", "-", "-"),
        }
    )
    target = pd.DataFrame(
        {"name": ("L5_IT", "L5_IT"), "strand": ("+", "-")}
    )

    actual = custom_model._semantic_bootstrap_track_indices(
        source,
        target,
        source_valid=(True,) * 4,
        source_valid_by_target=((True,) * 4,) * 2,
        seed=11,
    )

    assert {actual[0], actual[1]} == {1, 3}


def test_semantic_bootstrap_uses_seeded_fallback_without_match() -> None:
    source = pd.DataFrame(
        {
            "biosample_name": ("liver", "heart"),
            "strand": (".", "."),
        }
    )
    target = pd.DataFrame({"name": ("unknown_group",), "strand": (".",)})
    expected = custom_model._bootstrap_track_indices(
        (".", "."),
        (".",),
        source_valid=(True, True),
        source_valid_by_target=((True, True),),
        seed=17,
    )

    actual = custom_model._semantic_bootstrap_track_indices(
        source,
        target,
        source_valid=(True, True),
        source_valid_by_target=((True, True),),
        seed=17,
    )

    assert actual == expected


def test_pretrained_bootstrap_unwraps_target_output_metadata(monkeypatch) -> None:
    output_type = dna_output.OutputType.RNA_SEQ
    organism = dna_model.Organism.HOMO_SAPIENS

    class SourceMetadata:
        def __init__(self) -> None:
            self.frame = pd.DataFrame({"strand": ("+", "-")})
            self.padding = {output_type: jnp.asarray((False, False))}

        def get(self, requested):
            return self.frame if requested == output_type else None

    target_metadata = metadata_lib.AlphaGenomeOutputMetadata(
        rna_seq=pd.DataFrame({"strand": ("+", "-")})
    )
    monkeypatch.setattr(
        custom_model,
        "_resolve_user_metadata",
        lambda **_kwargs: {organism: target_metadata},
    )
    params = {
        "alphagenome/head/rna_seq/resolution_128/multi_organism_linear": {
            "w": jnp.asarray([[[1.0, 2.0]]]),
        },
        "head/study_rna/resolution_128/multi_organism_linear": {
            "w": jnp.zeros((1, 1, 2), dtype=jnp.float32),
        },
    }

    result = custom_model._initialize_heads_from_pretrained_bootstrap(
        params,
        head_names=("study_rna",),
        head_configs={"study_rna": SimpleNamespace(output_type=output_type)},
        pretrained_metadata={organism: SourceMetadata()},
    )

    actual = result["head/study_rna/resolution_128/multi_organism_linear"]["w"]
    assert set(actual.reshape(-1).tolist()) == {1.0, 2.0}


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
