import pytest

from scripts.eqtl.benchmark import _track_indices

ZEMKE2023_RNA = [
    "ASC", "Endo", "L2_3_IT", "L4_5_IT", "L5_6_NP", "L5_ET", "L5_IT", "L6_CT", "L6_IT", "L6_IT_CAR3",
    "L6b", "LAMP5", "MGC", "ODC", "OPC", "PVALB", "SNCG", "SST", "VIP", "VLMC",
]


def test_track_indices_follow_head_labels():
    indices = _track_indices(ZEMKE2023_RNA)
    named = {group: [ZEMKE2023_RNA[i] for i in positions] for group, positions in indices.items()}
    assert named["MGC"] == ["MGC"]
    assert named["ODC"] == ["ODC"]
    assert named["OPC"] == ["OPC"]
    assert named["INH"] == ["LAMP5", "PVALB", "SNCG", "SST", "VIP"]
    assert "L5_IT" not in named["EXC"] and "L6_IT" in named["EXC"]


def test_track_indices_reject_missing_labels():
    with pytest.raises(ValueError, match="L6_IT"):
        _track_indices([label for label in ZEMKE2023_RNA if label != "L6_IT"])
