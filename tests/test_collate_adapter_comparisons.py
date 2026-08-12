import json
from pathlib import Path

from scripts.v0data.collate_adapter_comparisons import (
    _variant_run_identity,
    collate,
    collate_variants,
)


def _write_metrics(root: Path, run: str, valid_r: float, test_r: float) -> None:
    run_dir = root / run
    run_dir.mkdir()
    record = {
        "epoch": 1,
        "global_step": 10,
        "metrics": {
            "valid": {"atac": {"differential_pearson_r": valid_r}},
            "test": {"atac": {"differential_pearson_r": test_r}},
        },
    }
    (run_dir / "metrics.jsonl").write_text(json.dumps(record) + "\n")


def test_variant_identity_preserves_dataset_strategy_and_variant() -> None:
    assert _variant_run_identity("hda-joint_lora_locon_geneonly_corrw1_screen") == (
        "hda-joint",
        "lora+locon",
        "geneonly_corrw1_screen",
    )
    assert _variant_run_identity("liu-hdma_lora_depth10m_full") == (
        "liu-hdma",
        "lora",
        "depth10m_full",
    )


def test_variant_identity_excludes_technical_runs() -> None:
    assert _variant_run_identity("hda-joint_lora_gradnorm") is None
    assert _variant_run_identity("hda-joint_lora_geneonly_smoke") is None


def test_canonical_and_variant_collation_are_disjoint(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "study_lora", 0.7, 0.6)
    _write_metrics(tmp_path, "study_lora_locon", 0.8, 0.7)
    _write_metrics(tmp_path, "study_lora_atac_nzmean_screen", 0.75, 0.65)
    _write_metrics(tmp_path, "study_lora_gradnorm", 0.1, 0.1)

    canonical = collate(tmp_path)
    variants = collate_variants(tmp_path)

    assert {(run["strategy"], run["dataset"]) for run in canonical["runs"]} == {
        ("lora", "study"),
        ("lora+locon", "study"),
    }
    assert [(run["variant"], run["selection_mean_valid_r"]) for run in variants["runs"]] == [
        ("atac_nzmean_screen", 0.75)
    ]
