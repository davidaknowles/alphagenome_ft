from pathlib import Path


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
