import numpy as np

from alphagenome_ft.finetune.metrics import r2_metrics, select_prediction_for_targets


def test_shared_r2_metrics_are_one_for_perfect_predictions():
    targets = np.square(np.arange(24, dtype=np.float32)).reshape(2, 3, 4)
    metrics = r2_metrics(targets.copy(), targets)

    assert metrics["r2_global"] == 1.0
    assert metrics["r2_over_loci"] == 1.0
    assert metrics["r2_over_cell_types"] == 1.0
    np.testing.assert_allclose(metrics["differential_pearson_r"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(metrics["double_centered_r"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(metrics["double_centered_r2"], 1.0, rtol=1e-6)


def test_shared_r2_metrics_accept_prediction_mapping():
    targets = np.square(np.arange(24, dtype=np.float32)).reshape(2, 3, 4)
    prediction = {"predictions_1bp": targets.copy()}

    selected = select_prediction_for_targets(prediction, targets)
    metrics = r2_metrics(prediction, targets)

    np.testing.assert_array_equal(selected, targets)
    assert metrics["r2_global"] == 1.0


def test_shared_differential_pearson_double_centers_loci_and_tracks():
    targets = np.square(np.arange(2 * 4 * 3, dtype=np.float32)).reshape(2, 4, 3)
    locus_offset = np.linspace(-3.0, 4.0, targets.shape[1], dtype=np.float32).reshape(1, -1, 1)
    track_offset = np.array([10.0, -5.0, 2.5], dtype=np.float32).reshape(1, 1, -1)
    prediction = targets + locus_offset + track_offset

    metrics = r2_metrics(prediction, targets)

    np.testing.assert_allclose(metrics["differential_pearson_r"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(metrics["double_centered_r"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(metrics["double_centered_r2"], 1.0, rtol=1e-6)


def test_double_centered_r2_is_squared_differential_correlation():
    targets = np.square(np.arange(2 * 4 * 3, dtype=np.float32)).reshape(2, 4, 3)
    prediction = targets[..., ::-1]

    metrics = r2_metrics(prediction, targets)

    np.testing.assert_allclose(
        metrics["double_centered_r2"],
        metrics["differential_pearson_r"] ** 2,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["double_centered_r"],
        metrics["differential_pearson_r"],
        rtol=1e-6,
    )
