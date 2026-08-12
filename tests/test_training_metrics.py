from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

from alphagenome_ft.finetune.train import (
    _double_centered_correlation_loss,
    _weighted_head_loss_sum,
    _finalize_r2_stats,
    _flatten_valid_metrics,
    _gene_expression_prediction,
    _r2_stats,
    _restore_optimizer_state,
    _save_optimizer_state,
)


def test_flatten_valid_metrics_builds_joint_selection_mean():
    flattened = _flatten_valid_metrics(
        {
            "atac": {"loss": 3.0, "differential_pearson_r": 0.8},
            "rna": {"loss": 1.0, "differential_pearson_r": 0.6},
        }
    )

    assert flattened["atac"] == 3.0
    assert flattened["rna/differential_pearson_r"] == 0.6
    np.testing.assert_allclose(flattened["mean/differential_pearson_r"], 0.7)


def test_weighted_head_loss_sum_rebalances_objective():
    total = _weighted_head_loss_sum(
        {"atac": jnp.asarray(3.0), "rna": jnp.asarray(2.0)},
        {
            "atac": SimpleNamespace(loss_weight=1.0),
            "rna": SimpleNamespace(loss_weight=5.0),
        },
    )

    np.testing.assert_allclose(total, 13.0)


def test_optimizer_state_roundtrip(tmp_path):
    import optax

    params = {"weight": jnp.asarray([1.0, 2.0])}
    optimizer = optax.adamw(1e-3)
    state = optimizer.init(params)
    gradients = {"weight": jnp.asarray([0.25, -0.5])}
    _, state = optimizer.update(gradients, state, params)
    path = tmp_path / "optimizer_state"

    _save_optimizer_state(path, state)
    restored = _restore_optimizer_state(path, optimizer.init(params))

    def assert_same_state(actual_state, expected_state):
        for actual, expected in zip(
            jax.tree_util.tree_leaves(actual_state),
            jax.tree_util.tree_leaves(expected_state),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)

    assert_same_state(restored, state)

    _, newer_state = optimizer.update(gradients, state, params)
    _save_optimizer_state(path, newer_state)
    restored_newer = _restore_optimizer_state(path, optimizer.init(params))
    assert_same_state(restored_newer, newer_state)


def test_double_centered_correlation_loss_matches_metric_invariances():
    targets = jnp.square(jnp.arange(24, dtype=jnp.float32)).reshape(2, 4, 3)
    locus_offset = jnp.arange(4, dtype=jnp.float32).reshape(1, 4, 1)
    track_offset = jnp.asarray([3.0, -2.0, 5.0]).reshape(1, 1, 3)
    predictions = {"predictions_1bp": targets + locus_offset + track_offset}

    loss = _double_centered_correlation_loss(predictions, targets)

    np.testing.assert_allclose(loss, 0.0, atol=1e-6)


def test_double_centered_correlation_loss_ignores_masked_rows():
    targets = jnp.square(jnp.arange(18, dtype=jnp.float32)).reshape(1, 6, 3)
    predictions = targets.at[:, -2:, :].set(-1000.0)
    mask = jnp.asarray([[True, True, True, True, False, False]])

    loss = _double_centered_correlation_loss(predictions, targets, mask)

    np.testing.assert_allclose(loss, 0.0, atol=1e-6)


def test_double_centered_correlation_loss_has_finite_gradient_without_variance():
    prediction = jnp.ones((2, 4, 3), dtype=jnp.float32)
    targets = jnp.ones((2, 4, 3), dtype=jnp.float32)

    loss, gradient = jax.value_and_grad(_double_centered_correlation_loss)(prediction, targets)

    assert loss == 0
    assert np.all(np.isfinite(gradient))
    np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


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


def test_r2_stats_sum_pool_targets_for_lower_resolution_prediction():
    targets = jnp.arange(512, dtype=jnp.float32).reshape(1, 256, 2)
    prediction = {"predictions_128bp": targets.reshape(1, 2, 128, 2).sum(axis=2)}

    stats = jax.tree_util.tree_map(np.asarray, _r2_stats(prediction, targets))
    metrics = _finalize_r2_stats(stats)

    assert np.isclose(metrics["r2_global"], 1.0)


def test_masked_r2_ignores_padded_genes():
    targets = jnp.asarray([[[1.0, 3.0], [2.0, 4.0], [99.0, 99.0]]])
    predictions = jnp.asarray([[[1.0, 3.0], [2.0, 4.0], [-99.0, -99.0]]])

    stats = jax.tree_util.tree_map(
        np.asarray,
        _r2_stats(predictions, targets, jnp.asarray([[True, True, False]])),
    )
    metrics = _finalize_r2_stats(stats)

    assert metrics["r2_global"] == 1.0
    assert metrics["r2_over_loci"] == 1.0


def test_gene_expression_prediction_selects_strand_and_exon_fraction():
    prediction = {
        "predictions_128bp": jnp.asarray([[[8.0, 80.0, 4.0, 40.0], [16.0, 160.0, 2.0, 20.0]]])
    }
    batch = {
        "gene_weights_rna": jnp.asarray([[[0.5, 0.25], [1.0, 0.5]]]),
        "gene_strands_rna": jnp.asarray([[0, 1]]),
    }

    result = np.asarray(_gene_expression_prediction(prediction, batch, "rna"))

    np.testing.assert_allclose(result[0, 0], [20.0, 4.0])
    np.testing.assert_allclose(result[0, 1], [100.0, 20.0])
