import json
from pathlib import Path

from scripts.v0data.audit_adapter_coverage import audit_coverage, render_markdown


def _write_epoch(root: Path, run: str, epoch: int) -> None:
    path = root / run / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"epoch": epoch, "metrics": {"valid": {"head": {}}}}) + "\n")


def _write_native_evaluation(
    root: Path,
    *,
    dataset: str,
    species: str,
    strategy: str,
) -> None:
    strategy_name = strategy.replace("+", "_")
    if dataset == "zemke2023_joint":
        run = f"zemke2023_{species}_{strategy_name}_joint_epoch1_eval"
    else:
        run = f"johansen_joint_{strategy_name}_{species}_eval"
    path = root / run / "evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_epoch": 1,
                "metrics": {"valid": {"head": {"differential_pearson_r": 0.5}}},
            }
        )
        + "\n"
    )


def test_audit_coverage_distinguishes_matched_and_missing_runs(tmp_path: Path) -> None:
    _write_epoch(tmp_path, "complete_lora", 1)
    _write_epoch(tmp_path, "complete_lora", 2)
    _write_epoch(tmp_path, "complete_lora_locon", 1)
    _write_epoch(tmp_path, "missing_lora", 1)

    result = audit_coverage(tmp_path, expected_datasets=("complete", "missing"))

    complete, missing = result["datasets"]
    assert complete["latest_lora_epoch"] == 2
    assert complete["latest_lora_locon_epoch"] == 1
    assert complete["highest_matched_epoch"] == 1
    assert complete["status"] == "matched result available"
    assert missing["highest_matched_epoch"] is None
    assert missing["status"] == "missing lora+locon"
    assert "does not imply that early stopping completed" in render_markdown(result)


def test_audit_coverage_excludes_superseded_johansen_checkpoint(tmp_path: Path) -> None:
    _write_epoch(tmp_path, "johansen_joint_lora_locon", 1)

    result = audit_coverage(tmp_path, expected_datasets=("johansen_joint",))

    assert result["datasets"][0]["status"] == "missing lora, lora+locon"


def test_audit_coverage_uses_corrected_reconstructed_run_names(tmp_path: Path) -> None:
    _write_epoch(tmp_path, "liu-hdma_lora", 1)
    _write_epoch(tmp_path, "liu-hdma_lora_geneonly_corrw1", 1)
    _write_epoch(tmp_path, "liu-hdma_lora_locon_geneonly_corrw1", 1)

    result = audit_coverage(tmp_path, expected_datasets=("liu-hdma",))

    assert result["datasets"][0]["highest_matched_epoch"] == 1


def test_primary_cross_species_study_requires_every_native_evaluation(tmp_path: Path) -> None:
    _write_epoch(tmp_path, "johansen_joint_lora_rawcount_geneonly_corrw1", 1)
    _write_epoch(tmp_path, "johansen_joint_lora_locon_rawcount_geneonly_corrw1", 1)
    for strategy in ("lora", "lora+locon"):
        for species in ("human", "macaque"):
            _write_native_evaluation(
                tmp_path,
                dataset="johansen_joint",
                species=species,
                strategy=strategy,
            )

    studies = (
        {
            "study": "Johansen",
            "canonical_dataset": "johansen_joint",
            "native_species": ("human", "macaque", "marmoset"),
        },
    )
    result = audit_coverage(
        tmp_path,
        expected_datasets=("johansen_joint",),
        primary_studies=studies,
    )

    assert result["primary_studies"][0]["canonical_status"] == "matched result available"
    assert result["primary_studies"][0]["status"] == "missing native evaluations"
    assert result["primary_studies"][0]["missing_native_evaluations"] == {
        "lora": ["marmoset"],
        "lora+locon": ["marmoset"],
    }
    assert "## Primary studies" in render_markdown(result)
