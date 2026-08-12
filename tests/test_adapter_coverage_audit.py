import json
from pathlib import Path

from scripts.v0data.audit_adapter_coverage import audit_coverage, render_markdown


def _write_epoch(root: Path, run: str, epoch: int) -> None:
    path = root / run / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"epoch": epoch, "metrics": {"valid": {"head": {}}}}) + "\n")


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
