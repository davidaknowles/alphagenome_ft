import numpy as np

from scripts.v0data.audit_atac_depth_signal import summarize_relationship


def test_summarize_relationship_reports_depth_signal_correlations() -> None:
    values = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    result = summarize_relationship(
        ["low", "medium", "high"],
        values,
        np.asarray([1_000_000, 10_000_000, 100_000_000], dtype=np.float64),
    )

    assert result["num_tracks"] == 3
    assert result["groups_below_5m"] == 1
    assert result["groups_below_25m"] == 2
    assert len(result["tracks"]) == 3
    assert np.isfinite(result["spearman_log_depth_vs_log_rms"])
    assert np.isfinite(result["spearman_log_depth_vs_median_track_correlation"])
