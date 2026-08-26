import json

import numpy as np

from scripts.v0data.zemke2023_rna_reprocessing.audit_marmoset_test_outlier import (
    _matched_evaluation,
    target_leverage,
)


def test_target_leverage_attributes_an_outlier_to_excluded_region():
    values = np.asarray(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [100.0, 1.0],
            [3.0, 3.0],
        ]
    )

    result = target_leverage(
        values,
        chromosome_size=400,
        excluded_start=200,
        excluded_end=300,
    )

    assert result["excluded_bin_count"] == 1
    assert result["excluded_locus_variance_fraction"] > 0.7
    assert result["maximum_track_index"] == 0


def test_matched_evaluation_requires_shared_checkpoint_and_reports_delta(tmp_path):
    def write(name, rna_r):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "source_epoch": 4,
                    "source_global_step": 10,
                    "metrics": {
                        "test": {
                            "zemke2023_atac": {"differential_pearson_r": 0.6},
                            "zemke2023_rna": {"differential_pearson_r": rna_r},
                        }
                    },
                }
            )
        )
        return path

    result = _matched_evaluation(write("baseline.json", 0.15), write("excluded.json", 0.63))

    assert np.isclose(result["heads"]["zemke2023_rna"]["difference"], 0.48)
    assert result["source_epoch"] == 4
