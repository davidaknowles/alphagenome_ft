import json
from pathlib import Path

from scripts.v0data.prepare_joint_metric_aligned_config import (
    CORRELATION_WEIGHTS,
    prepare_config,
)


def test_prepare_config_applies_every_dataset_policy(tmp_path: Path) -> None:
    datasets = []
    for dataset, policy in CORRELATION_WEIGHTS.items():
        target_path = tmp_path / "source" / dataset / "targets.json"
        target_path.parent.mkdir(parents=True)
        target_path.write_text(
            json.dumps(
                {
                    "heads": [
                        {"id": head_id, "double_centered_correlation_loss_weight": 0.0}
                        for head_id in policy
                    ]
                    + [{"id": f"{dataset}_unchanged", "loss_weight": 1.0}]
                }
            )
        )
        datasets.append(
            {
                "name": dataset,
                "sources": [{"name": "human", "targets_config": str(target_path)}],
            }
        )

    result = prepare_config(
        {"datasets": datasets},
        source_path=tmp_path / "source" / "datasets.json",
        output_dir=tmp_path / "output",
    )

    for dataset in result["datasets"]:
        copied_path = Path(dataset["sources"][0]["targets_config"])
        heads = {head["id"]: head for head in json.loads(copied_path.read_text())["heads"]}
        for head_id, weight in CORRELATION_WEIGHTS[dataset["name"]].items():
            assert heads[head_id]["double_centered_correlation_loss_weight"] == weight
        assert heads[f"{dataset['name']}_unchanged"] == {
            "id": f"{dataset['name']}_unchanged",
            "loss_weight": 1.0,
        }
