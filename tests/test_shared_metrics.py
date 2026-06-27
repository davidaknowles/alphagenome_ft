import numpy as np

from alphagenome_ft.finetune.metrics import r2_metrics, select_prediction_for_targets


def test_shared_r2_metrics_are_one_for_perfect_predictions():
    targets = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    metrics = r2_metrics(targets.copy(), targets)

    assert metrics["r2_global"] == 1.0
    assert metrics["r2_over_loci"] == 1.0
    assert metrics["r2_over_cell_types"] == 1.0


def test_shared_r2_metrics_accept_prediction_mapping():
    targets = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    prediction = {"predictions_1bp": targets.copy()}

    selected = select_prediction_for_targets(prediction, targets)
    metrics = r2_metrics(prediction, targets)

    np.testing.assert_array_equal(selected, targets)
    assert metrics["r2_global"] == 1.0
