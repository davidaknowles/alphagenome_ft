import json

import pytest

from scripts.allen_atac_reprocessing.collate_split_half_reliability import collate, markdown


def _payload(species: str, split_half: float) -> dict[str, object]:
    return {
        "species": species,
        "chromosome": "chr9",
        "signal": "coverage",
        "groups": 10,
        "split_half_double_centered_r": split_half,
        "full_target_reliability_estimate": 0.8,
        "model_correlation_ceiling_estimate": 0.9,
    }


def test_collate_orders_species_and_renders_separate_rows(tmp_path):
    macaque = tmp_path / "macaque.json"
    human = tmp_path / "human.json"
    macaque.write_text(json.dumps(_payload("macaque", 0.7)))
    human.write_text(json.dumps(_payload("human", 0.6)))

    rows = collate([macaque, human])

    assert [row["species"] for row in rows] == ["human", "macaque"]
    assert "| human | chr9 | 10 | 0.6000 | 0.8000 | 0.9000 |" in markdown(rows)


def test_collate_rejects_noncoverage_signal(tmp_path):
    path = tmp_path / "input.json"
    payload = _payload("human", 0.6)
    payload["signal"] = "insertion"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="expected 'coverage'"):
        collate([path])
