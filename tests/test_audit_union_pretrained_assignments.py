from scripts.v0data.audit_union_pretrained_assignments import render_markdown


def test_render_assignment_audit_summarizes_routes() -> None:
    result = {
        "initializer": "semantic_neural_accessibility_bootstrap",
        "routes": [
            {
                "dataset": "study",
                "source": "human",
                "head": "study_rna",
                "pretrained_assay": "rna_seq",
                "target_channels": 20,
                "neural_target_channels": 18,
                "unique_source_channels": 13,
                "maximum_source_reuse": 2,
            }
        ]
    }

    rendered = render_markdown(result)

    assert "`study_rna`" in rendered
    assert "| 20 | 18 | 13 | 2 |" in rendered
    assert "prefers matching anatomy" in rendered


def test_render_assignment_audit_labels_shuffled_baseline() -> None:
    rendered = render_markdown(
        {"initializer": "neural_accessibility_bootstrap", "routes": []}
    )

    assert "without matching target labels" in rendered
