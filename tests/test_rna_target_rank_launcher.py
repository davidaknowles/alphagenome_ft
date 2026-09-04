from pathlib import Path

from scripts.v0data.audit_rna_target_rank import render_markdown


def test_rank_audit_covers_all_retained_direct_rna_targets() -> None:
    script = Path("scripts/v0data/run_rna_target_rank_audit.sh").read_text()

    for dataset in (
        "HDA",
        "Liu",
        "Johansen-human",
        "Johansen-macaque",
        "Johansen-marmoset",
        "Zemke2023-human",
        "Zemke2023-macaque",
        "Zemke2023-marmoset",
        "Zemke2023-mouse",
        "Zemke2024",
    ):
        assert f"--dataset {dataset}=" in script


def test_rank_audit_renderer_marks_unavailable_rank_32() -> None:
    result = {
        "datasets": {
            "small": {
                scale: {
                    "observations": 100,
                    "tracks": 20,
                    "entropy_effective_rank": 4.0,
                    "rank_for_correlation": {"0.8": 3},
                    "rank_correlation_ceiling": {
                        "1": 0.5,
                        "4": 0.8,
                        "8": 0.9,
                        "16": 0.99,
                    },
                }
                for scale in ("raw_cpm", "log1p_cpm")
            }
        }
    }

    rendered = render_markdown(result)

    assert "| small | 100 | 20 | 4.00 | 3 | 0.5000 | 0.8000 | 0.9000 | 0.9900 | - |" in rendered
