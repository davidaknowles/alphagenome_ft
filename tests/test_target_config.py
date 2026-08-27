import dataclasses
from pathlib import Path

import pytest

from alphagenome.models import dna_client, dna_output

from alphagenome_ft.finetune.config import (
    HeadSpec,
    TrackInfo,
    _build_track_metadata,
    head_layout_signature,
    load_targets_config,
    prepare_head_specs,
    validate_head_specs,
)
from alphagenome_ft.custom_heads import FactorizedGenomeTracksHeadConfig


def test_track_metadata_preserves_strands(tmp_path: Path):
    tracks = [
        TrackInfo(name="cell (+)", path=tmp_path / "plus.bw", strand="+"),
        TrackInfo(name="cell (-)", path=tmp_path / "minus.bw", strand="-"),
    ]

    metadata = _build_track_metadata(
        tracks,
        dna_client.Organism.HOMO_SAPIENS,
        dna_output.OutputType.RNA_SEQ,
    )[dna_client.Organism.HOMO_SAPIENS]

    assert metadata.rna_seq["strand"].tolist() == ["+", "-"]
    assert metadata.strand_reindexing[dna_output.OutputType.RNA_SEQ].tolist() == [1, 0]


def test_head_layout_signature_can_ignore_source_specific_ids(tmp_path: Path):
    tracks = (TrackInfo(name="cell", path=tmp_path / "track.bw"),)
    human = HeadSpec(head_id="rna_human", source="predefined", kind="rna_seq", tracks=tracks)
    macaque = dataclasses.replace(human, head_id="rna_macaque")

    assert head_layout_signature((human,)) != head_layout_signature((macaque,))
    assert head_layout_signature(
        (human,), include_head_ids=False
    ) == head_layout_signature((macaque,), include_head_ids=False)


def test_load_targets_config_resolves_target_transform_path(tmp_path: Path):
    config = load_targets_config(
        {
            "heads": [
                {
                    "id": "example",
                    "source": "predefined",
                    "targets": [{"path": "track.bw"}],
                    "target_transform": {"path": "transform.json"},
                }
            ]
        },
        base_dir=tmp_path,
    )

    head = config["heads"][0]
    assert head["targets"][0]["path"] == str(tmp_path / "track.bw")
    assert head["target_transform"]["path"] == str(tmp_path / "transform.json")


def test_prepare_head_specs_parses_head_loss_weight(tmp_path: Path):
    track_path = tmp_path / "track.bw"
    track_path.touch()
    specs = prepare_head_specs(
        {
            "heads": [
                {
                    "id": "example_atac",
                    "source": "predefined",
                    "kind": "atac",
                    "loss_weight": 5.0,
                    "targets": [{"path": str(track_path)}],
                }
            ]
        },
        organism="HOMO_SAPIENS",
    )

    assert specs[0].loss_weight == 5.0


def test_prepare_head_specs_parses_row_correlation_weight(tmp_path: Path):
    track_path = tmp_path / "track.bw"
    track_path.touch()
    specs = prepare_head_specs(
        {
            "heads": [
                {
                    "id": "example_atac",
                    "source": "predefined",
                    "kind": "atac",
                    "row_centered_correlation_loss_weight": 2.0,
                    "targets": [{"path": str(track_path)}],
                }
            ]
        },
        organism="HOMO_SAPIENS",
    )

    assert specs[0].row_centered_correlation_loss_weight == 2.0


def test_prepare_head_specs_parses_factorized_rna_output(tmp_path: Path):
    tracks = []
    for index in range(20):
        path = tmp_path / f"track{index}.bw"
        path.touch()
        tracks.append({"path": str(path), "strand": "+" if index % 2 == 0 else "-"})
    specs = prepare_head_specs(
        {
            "heads": [
                {
                    "id": "example_rna",
                    "source": "predefined",
                    "kind": "rna_seq",
                    "output_rank": 16,
                    "targets": tracks,
                }
            ]
        },
        organism="HOMO_SAPIENS",
    )

    assert specs[0].output_rank == 16
    assert isinstance(specs[0].config, FactorizedGenomeTracksHeadConfig)
    assert specs[0].config.output_rank == 16


def test_prepare_head_specs_rejects_factorized_atac_output(tmp_path: Path):
    track = tmp_path / "track.bw"
    track.touch()
    with pytest.raises(ValueError, match="only for RNA-seq"):
        prepare_head_specs(
            {
                "heads": [
                    {
                        "id": "example_atac",
                        "source": "predefined",
                        "kind": "atac",
                        "output_rank": 1,
                        "targets": [{"path": str(track)}, {"path": str(track)}],
                    }
                ]
            },
            organism="HOMO_SAPIENS",
        )


@pytest.mark.parametrize("weight", [0.0, -1.0, float("inf"), float("nan")])
def test_prepare_head_specs_rejects_invalid_head_loss_weight(tmp_path: Path, weight: float):
    track_path = tmp_path / "track.bw"
    track_path.touch()
    with pytest.raises(ValueError, match="finite and positive"):
        prepare_head_specs(
            {
                "heads": [
                    {
                        "id": "example_atac",
                        "source": "predefined",
                        "kind": "atac",
                        "loss_weight": weight,
                        "targets": [{"path": str(track_path)}],
                    }
                ]
            },
            organism="HOMO_SAPIENS",
        )


@pytest.mark.parametrize("weight", [-1.0, float("inf"), float("nan")])
def test_prepare_head_specs_rejects_invalid_row_correlation_weight(
    tmp_path: Path, weight: float
):
    track_path = tmp_path / "track.bw"
    track_path.touch()
    with pytest.raises(ValueError, match="Row-centered correlation loss weight"):
        prepare_head_specs(
            {
                "heads": [
                    {
                        "id": "example_atac",
                        "source": "predefined",
                        "kind": "atac",
                        "row_centered_correlation_loss_weight": weight,
                        "targets": [{"path": str(track_path)}],
                    }
                ]
            },
            organism="HOMO_SAPIENS",
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("double_centered_correlation_loss_weight", "double-centered"),
        ("row_centered_correlation_loss_weight", "row-centered"),
    ],
)
def test_validate_head_specs_rejects_invalid_correlation_weight(
    tmp_path: Path, field: str, message: str
):
    track_path = tmp_path / "track.bw"
    track_path.touch()
    spec = prepare_head_specs(
        {
            "heads": [
                {
                    "id": "example_atac",
                    "source": "predefined",
                    "kind": "atac",
                    "targets": [{"path": str(track_path)}],
                }
            ]
        },
        organism="HOMO_SAPIENS",
    )[0]

    with pytest.raises(ValueError, match=message):
        validate_head_specs([dataclasses.replace(spec, **{field: float("nan")})])
