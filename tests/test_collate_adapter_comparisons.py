import json
from pathlib import Path

from scripts.v0data.collate_adapter_comparisons import (
    _run_identity,
    _variant_run_identity,
    canonical_run_directory,
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


def _append_metrics(root: Path, run: str, epoch: int, valid_r: float, test_r: float) -> None:
    record = {
        "epoch": epoch,
        "global_step": epoch * 10,
        "metrics": {
            "valid": {"atac": {"differential_pearson_r": valid_r}},
            "test": {"atac": {"differential_pearson_r": test_r}},
        },
    }
    with (root / run / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


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


def test_corrected_reconstructed_runs_are_canonical() -> None:
    assert _run_identity("joint_all_nonencode_lora_provisional") == (
        "joint-all-nonencode",
        "lora",
    )
    assert _variant_run_identity("joint_all_nonencode_lora_provisional") is None
    assert canonical_run_directory("liu-hdma", "lora") == (
        "liu-hdma_lora_geneonly_corrw1"
    )
    assert _run_identity("liu-hdma_lora_geneonly_corrw1") == ("liu-hdma", "lora")
    assert _run_identity("johansen_joint_lora_locon_rawcount_geneonly_corrw1") == (
        "johansen_joint",
        "lora+locon",
    )
    assert _run_identity("liu-hdma_lora") is None
    assert _run_identity("johansen_joint_lora_locon") is None
    assert _variant_run_identity("liu-hdma_lora_geneonly_corrw1") is None
    assert _variant_run_identity("liu-hdma_lora") == (
        "liu-hdma",
        "lora",
        "legacy_exon_plus_gene",
    )


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


def test_variant_comparison_uses_highest_common_epoch(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "study_lora_rank16_screen", 0.6, 0.5)
    _append_metrics(tmp_path, "study_lora_rank16_screen", 2, 0.9, 0.8)
    _write_metrics(tmp_path, "study_lora_locon_rank16_screen", 0.7, 0.6)

    result = collate_variants(tmp_path)

    assert {run["matched_epoch"] for run in result["matched_runs"]} == {1}
    assert {
        (run["strategy"], run["heads"][0]["valid_r"])
        for run in result["matched_runs"]
    } == {("lora", 0.6), ("lora+locon", 0.7)}
    selected_lora = next(run for run in result["runs"] if run["strategy"] == "lora")
    assert selected_lora["selected_epoch"] == 2


def test_superseded_johansen_checkpoint_is_excluded(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "johansen_joint_lora_locon", 0.8, 0.8)

    assert collate(tmp_path)["runs"] == []


def test_corrected_reconstructed_runs_enter_canonical_comparison(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "liu-hdma_lora_geneonly_corrw1", 0.7, 0.6)
    _write_metrics(tmp_path, "liu-hdma_lora_locon_geneonly_corrw1", 0.8, 0.7)

    result = collate(tmp_path)

    assert {(run["dataset"], run["strategy"]) for run in result["runs"]} == {
        ("liu-hdma", "lora"),
        ("liu-hdma", "lora+locon"),
    }
    assert len(result["matched_runs"]) == 2


def test_matched_comparison_uses_highest_common_epoch(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "study_lora", 0.6, 0.5)
    _append_metrics(tmp_path, "study_lora", 2, 0.9, 0.8)
    _write_metrics(tmp_path, "study_lora_locon", 0.7, 0.6)

    result = collate(tmp_path)

    assert {run["matched_epoch"] for run in result["matched_runs"]} == {1}
    assert {
        (run["strategy"], run["heads"][0]["valid_r"])
        for run in result["matched_runs"]
    } == {("lora", 0.6), ("lora+locon", 0.7)}
    selected_lora = next(run for run in result["runs"] if run["strategy"] == "lora")
    assert selected_lora["selected_epoch"] == 2


def test_canonical_collation_rejects_nonfinite_head_correlation(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "study_lora", 0.7, 0.6)
    _write_metrics(tmp_path, "study_lora_locon", 0.8, 0.7)
    path = tmp_path / "study_lora_locon" / "metrics.jsonl"
    record = json.loads(path.read_text())
    record["metrics"]["valid"]["atac"]["differential_pearson_r"] = None
    path.write_text(json.dumps(record) + "\n")

    result = collate(tmp_path)

    assert [(run["strategy"], run["selected_epoch"]) for run in result["runs"]] == [
        ("lora", 1)
    ]
    assert result["matched_runs"] == []
