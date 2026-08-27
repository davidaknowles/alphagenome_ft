import json
from pathlib import Path

import pytest

from scripts.v0data.prepare_source_specific_joint_heads import prepare_config


def _manifest(path: Path, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "heads": [
                    {"id": f"{prefix}_atac", "kind": "atac", "targets": []},
                    {"id": f"{prefix}_rna", "kind": "rna_seq", "targets": []},
                ]
            }
        )
    )


def test_prepare_config_renames_selected_native_source_heads(tmp_path: Path) -> None:
    manifests = {}
    for species in ("human", "marmoset"):
        path = tmp_path / "source" / species / "targets.json"
        _manifest(path, "allen")
        manifests[species] = path
    source_path = tmp_path / "source" / "datasets.json"
    payload = {
        "sampling_strategy": "equal_sources",
        "datasets": [
            {
                "name": "johansen2025",
                "sources": [
                    {"name": species, "targets_config": str(path)}
                    for species, path in manifests.items()
                ],
            },
            {
                "name": "hda",
                "sources": [{"name": "human", "targets_config": "unchanged.json"}],
            },
        ],
        "objective_variant": {
            "double_centered_correlation_loss_weights": {
                "johansen2025": {"allen_rna": 1.0}
            }
        },
    }

    result = prepare_config(
        payload,
        source_path=source_path,
        output_dir=tmp_path / "output",
        dataset_names=("johansen2025",),
    )

    johansen = result["datasets"][0]
    assert johansen["source_specific_heads"] is True
    assert result["datasets"][1] == payload["datasets"][1]
    for source in johansen["sources"]:
        manifest = json.loads(Path(source["targets_config"]).read_text())
        assert [head["id"] for head in manifest["heads"]] == [
            f"allen_atac_{source['name']}",
            f"allen_rna_{source['name']}",
        ]
    assert result["objective_variant"]["double_centered_correlation_loss_weights"][
        "johansen2025"
    ] == {"allen_rna_human": 1.0, "allen_rna_marmoset": 1.0}


def test_prepare_config_requires_equal_source_exposure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="equal_sources"):
        prepare_config(
            {"datasets": []},
            source_path=tmp_path / "datasets.json",
            output_dir=tmp_path / "output",
            dataset_names=(),
        )
