import numpy as np
import jax
import jax.numpy as jnp

from alphagenome_ft.finetune.train import _finalize_r2_stats, _r2_stats


def test_r2_stats_are_one_for_perfect_predictions():
    targets = jnp.asarray([[[1.0, 3.0], [2.0, 4.0]]])
    predictions = {"predictions_1bp": targets}

    stats = jax.tree_util.tree_map(np.asarray, _r2_stats(predictions, targets))
    metrics = _finalize_r2_stats(stats)

    assert metrics["r2_global"] == 1.0
    assert metrics["r2_over_loci"] == 1.0
    assert metrics["r2_over_cell_types"] == 1.0


def test_double_centered_r2_is_one_for_nonadditive_perfect_predictions():
    targets = jnp.square(jnp.arange(18, dtype=jnp.float32)).reshape(2, 3, 3)
    predictions = {"predictions_1bp": targets}

    stats = jax.tree_util.tree_map(np.asarray, _r2_stats(predictions, targets))
    metrics = _finalize_r2_stats(stats)

    np.testing.assert_allclose(metrics["double_centered_r2"], 1.0, rtol=1e-6)


def test_r2_stats_track_loci_and_cell_type_axes_separately():
    targets = jnp.asarray([[[1.0, 3.0], [2.0, 4.0]]])
    predictions = {"predictions_1bp": jnp.asarray([[[1.0, 3.0], [2.0, 2.0]]])}

    stats = jax.tree_util.tree_map(np.asarray, _r2_stats(predictions, targets))
    metrics = _finalize_r2_stats(stats)

    assert metrics["r2_global"] < 1.0
    assert metrics["r2_over_loci"] < 1.0
    assert metrics["r2_over_cell_types"] < 1.0
