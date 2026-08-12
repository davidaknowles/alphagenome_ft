from scripts.v0data.audit_metric_targets import audit_metric_targets, render_markdown


def test_metric_target_audit_distinguishes_reached_below_and_missing() -> None:
    canonical = {
        "runs": [
            {
                "dataset": "study",
                "strategy": "lora",
                "selected_epoch": 2,
                "heads": [
                    {"head": "study_atac", "valid_r": 0.81, "test_r": 0.80},
                    {"head": "study_rna", "valid_r": 0.60, "test_r": 0.55},
                ],
            },
            {
                "dataset": "study",
                "strategy": "lora+locon",
                "selected_epoch": 2,
                "heads": [
                    {"head": "study_atac", "valid_r": 0.80, "test_r": 0.79},
                    {"head": "study_rna", "valid_r": 0.65, "test_r": 0.60},
                ],
            },
        ]
    }

    result = audit_metric_targets(
        canonical,
        threshold=0.8,
        expected_datasets=("study", "missing"),
    )

    rows = {(row["dataset"], row["modality"]): row for row in result["rows"]}
    assert rows[("study", "ATAC")]["status"] == "target reached"
    assert rows[("study", "RNA")]["strategy"] == "lora+locon"
    assert rows[("study", "RNA")]["status"] == "below target"
    assert rows[("missing", "ATAC")]["status"] == "missing evidence"
    assert "Missing evidence is distinct" in render_markdown(result)
