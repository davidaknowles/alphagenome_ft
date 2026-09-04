import json
from pathlib import Path

import pytest

from scripts.v0data.collate_joint_multidataset_evaluations import collate, render_markdown


def _write_evaluation(path: Path, epoch: int, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_epoch": epoch,
                "source_global_step": epoch * 100,
                "metrics": {
                    split: {
                        "study_atac": {
                            "differential_pearson_r": offset + split_offset
                        },
                        "study_rna": {
                            "differential_pearson_r": offset + split_offset - 0.1
                        },
                    }
                    for split, split_offset in (("valid", 0.0), ("test", 0.05))
                },
            }
        )
        + "\n"
    )


def test_collate_requires_and_reports_every_strategy_source(tmp_path: Path):
    config = tmp_path / "datasets.json"
    config.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "study",
                        "sources": [{"name": "human"}, {"name": "mouse"}],
                    }
                ]
            }
        )
    )
    root = tmp_path / "evaluations"
    for strategy, epoch, offset in (("lora", 2, 0.7), ("lora_locon", 3, 0.72)):
        for source in ("human", "mouse"):
            _write_evaluation(
                root / strategy / f"study_{source}" / "evaluation.json",
                epoch,
                offset,
            )

    result = collate(config, root)

    assert len(result["rows"]) == 8
    assert result["strategy_summaries"][0] == {
        "strategy": "lora",
        "native_sources": 2,
        "heads": 4,
        "mean_valid_r": pytest.approx(0.65),
        "mean_test_r": pytest.approx(0.70),
        "mean_atac_valid_r": pytest.approx(0.7),
        "mean_atac_test_r": pytest.approx(0.75),
        "mean_rna_valid_r": pytest.approx(0.6),
        "mean_rna_test_r": pytest.approx(0.65),
    }
    markdown = render_markdown(result)
    assert "| `study` | `mouse` | `lora+locon` | 3 | `study_rna` |" in markdown


def test_collate_rejects_missing_native_source_evaluation(tmp_path: Path):
    config = tmp_path / "datasets.json"
    config.write_text(
        json.dumps({"datasets": [{"name": "study", "sources": [{"name": "human"}]}]})
    )

    with pytest.raises(FileNotFoundError, match="lora/study_human/evaluation.json"):
        collate(config, tmp_path / "evaluations")


def test_collate_accepts_arbitrary_labeled_runs(tmp_path: Path):
    config = tmp_path / "datasets.json"
    config.write_text(
        json.dumps({"datasets": [{"name": "study", "sources": [{"name": "human"}]}]})
    )
    root = tmp_path / "evaluations"
    for run, epoch, offset in (("control", 4, 0.7), ("weighted", 5, 0.72)):
        _write_evaluation(root / run / "study_human" / "evaluation.json", epoch, offset)

    result = collate(
        config,
        root,
        runs={"control": "LoCon control", "weighted": "LoCon RNA weight 2"},
    )

    assert [row["strategy"] for row in result["rows"]] == [
        "LoCon control",
        "LoCon control",
        "LoCon RNA weight 2",
        "LoCon RNA weight 2",
    ]
    assert [summary["strategy"] for summary in result["strategy_summaries"]] == [
        "LoCon control",
        "LoCon RNA weight 2",
    ]


def test_collate_uses_manifest_kind_for_source_specific_head_ids(tmp_path: Path):
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps(
            {
                "heads": [
                    {"id": "allen_atac_human", "kind": "atac"},
                    {"id": "allen_rna_human", "kind": "rna_seq"},
                ]
            }
        )
    )
    config = tmp_path / "datasets.json"
    config.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "johansen2025",
                        "sources": [
                            {"name": "human", "targets_config": str(manifest)}
                        ],
                    }
                ]
            }
        )
    )
    root = tmp_path / "evaluations"
    for run, offset in (("lora", 0.7), ("lora_locon", 0.72)):
        path = root / run / "johansen2025_human" / "evaluation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "source_epoch": 2,
                    "source_global_step": 200,
                    "metrics": {
                        split: {
                            "allen_atac_human": {
                                "differential_pearson_r": offset + split_offset
                            },
                            "allen_rna_human": {
                                "differential_pearson_r": offset + split_offset - 0.1
                            },
                        }
                        for split, split_offset in (("valid", 0.0), ("test", 0.05))
                    },
                }
            )
        )

    result = collate(config, root)

    assert result["strategy_summaries"][0]["mean_atac_valid_r"] == pytest.approx(0.7)
    assert result["strategy_summaries"][0]["mean_rna_valid_r"] == pytest.approx(0.6)
