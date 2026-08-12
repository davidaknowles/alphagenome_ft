import json
from pathlib import Path

import pytest

from scripts.v0data.johansen_rna_reprocessing.collate_donor_reliability import (
    collate,
    render_markdown,
)


def _write(path: Path, species: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "species": species,
                "donors": 4,
                "groups": 47,
                "genes": 100,
                "groups_estimable_in_both_halves": 46,
                "raw_cpm_double_centered_r": 0.8,
                "raw_cpm_full_reliability_estimate": 8 / 9,
                "raw_cpm_model_correlation_ceiling_estimate": (8 / 9) ** 0.5,
                "log1p_cpm_double_centered_r": 0.7,
            }
        )
    )
    return path


def test_collate_orders_species_and_renders_ceiling(tmp_path: Path) -> None:
    paths = [_write(tmp_path / f"{species}.json", species) for species in ("marmoset", "human", "macaque")]

    result = collate(paths)

    assert [audit["species"] for audit in result["audits"]] == ["human", "macaque", "marmoset"]
    assert "0.9428" in render_markdown(result)


def test_collate_requires_every_species(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        collate([_write(tmp_path / "human.json", "human")])


def test_collate_accepts_explicit_species_order_and_title(tmp_path: Path) -> None:
    paths = [
        _write(tmp_path / f"{species}.json", species)
        for species in ("mouse", "human")
    ]
    result = collate(paths, ("human", "mouse"), "Auxiliary target only.")
    markdown = render_markdown(
        result,
        title="Zemke RNA reliability",
        qualification="Auxiliary target only.",
    )
    assert [audit["species"] for audit in result["audits"]] == ["human", "mouse"]
    assert result["qualification"] == "Auxiliary target only."
    assert markdown.startswith("# Zemke RNA reliability\n")
    assert "Auxiliary target only." in markdown
