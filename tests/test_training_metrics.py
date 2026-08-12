from contextlib import nullcontext
import functools
import json
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

from alphagenome_ft.finetune.train import (
    _double_centered_correlation_loss,
    _row_centered_correlation_loss,
    _weighted_head_loss_sum,
    _finalize_r2_stats,
    _flatten_valid_metrics,
    _gene_expression_prediction,
    _gradient_inner_product,
    _gradient_l2_norm,
    _is_lora_path,
    _r2_stats,
    _restore_optimizer_state,
    _save_gradient_diagnostics,
    _save_optimizer_state,
    train,
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


def test_gradient_norm_helpers_filter_parameter_paths():
    first = {"trunk": {"lora_a": jnp.asarray([3.0, 4.0]), "weight": jnp.asarray([100.0])}}
    second = {"trunk": {"lora_a": jnp.asarray([1.0, -2.0]), "weight": jnp.asarray([100.0])}}

    np.testing.assert_allclose(_gradient_l2_norm(first, _is_lora_path), 5.0)
    np.testing.assert_allclose(
        _gradient_inner_product(first, second, _is_lora_path),
        -5.0,
    )


def test_train_reports_per_head_gradient_norms_once(capsys, tmp_path):
    from alphagenome.models import dna_model as ag_dna_model

    class DummyDataModule:
        _batch_size = 1
        _drop_last = False
        _intervals = {
            split: [object(), object()] for split in ("train", "valid", "test")
        }

        def __init__(self):
            self.calls = {split: 0 for split in self._intervals}

        def iter_batches(self, split, seed=None, shuffle=True):
            del seed, shuffle
            self.calls[split] += 1
            sequence = np.arange(4, dtype=np.float32).reshape(1, 4, 1)
            for _ in range(2):
                yield {
                    "sequences": sequence,
                    "negative_strand_mask": np.asarray([False]),
                    "targets_atac": 2.0 * sequence,
                    "targets_rna": 3.0 * sequence,
                }

    class DummyModel:
        def __init__(self):
            self._params = {
                "trunk": {"lora_a": jnp.asarray(0.5)},
                "head": {
                    "atac": {"weight": jnp.asarray(1.0)},
                    "rna": {"weight": jnp.asarray(1.5)},
                },
            }
            self._state = {}
            self._device_context = nullcontext()
            self._metadata = {
                ag_dna_model.Organism.HOMO_SAPIENS: SimpleNamespace(
                    strand_reindexing=jnp.asarray([0], dtype=jnp.int32)
                )
            }

        def freeze_backbone(self):
            return None

        def create_loss_fn_for_head(self, head_name):
            del head_name

            def loss_fn(prediction, target):
                residual = prediction["predictions_1bp"] - target["targets"]
                return {"loss": jnp.mean(jnp.square(residual))}

            return loss_fn

        def _predict(
            self,
            params,
            state,
            sequences,
            organism_index,
            *,
            requested_outputs,
            negative_strand_mask,
            strand_reindexing,
        ):
            del state, organism_index, negative_strand_mask, strand_reindexing
            adapter = params["trunk"]["lora_a"]
            return {
                head_name: {
                    "predictions_1bp": sequences * adapter * params["head"][head_name]["weight"]
                }
                for head_name in requested_outputs
            }

    specs = [
        SimpleNamespace(
            head_id=head_name,
            target_transform_path=None,
            gene_supervision_path=None,
            coverage_loss_weight=1.0,
            gene_loss_weight=0.0,
            double_centered_correlation_loss_weight=0.0,
            row_centered_correlation_loss_weight=0.0,
            loss_weight=weight,
        )
        for head_name, weight in (("atac", 1.0), ("rna", 5.0))
    ]

    training_data = DummyDataModule()
    train(
        DummyModel(),
        training_data,
        specs,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        max_train_steps=2,
        heads_only=True,
        train_lora=True,
        eval_splits=("train",),
        prefetch_batches=0,
        report_head_gradient_norms=True,
    )

    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.startswith("Head gradient diagnostics:")]
    assert len(lines) == 1
    diagnostics = json.loads(lines[0].split(": ", 1)[1])
    for head_name in ("atac", "rna"):
        assert diagnostics["heads"][head_name]["adapter_gradient_norm"] > 0
        assert diagnostics["heads"][head_name]["head_gradient_norm"] > 0
    rna = diagnostics["heads"]["rna"]
    np.testing.assert_allclose(
        rna["weighted_adapter_gradient_norm"],
        5.0 * rna["adapter_gradient_norm"],
    )
    diagnostics_dir = tmp_path / "diagnostics"
    diagnostics_dir.mkdir()
    _save_gradient_diagnostics(
        diagnostics_dir,
        diagnostics,
        epoch=1,
        global_step=0,
    )
    persisted = json.loads((diagnostics_dir / "gradient_diagnostics.json").read_text())
    assert persisted["epoch"] == 1
    assert persisted["global_step_before_update"] == 0
    assert persisted["heads"] == diagnostics["heads"]

    evaluation_model = DummyModel()
    original_params = jax.tree_util.tree_map(np.asarray, evaluation_model._params)
    evaluation = train(
        evaluation_model,
        DummyDataModule(),
        specs,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        heads_only=True,
        train_lora=True,
        checkpoint_dir=tmp_path / "evaluation",
        eval_splits=("train",),
        prefetch_batches=0,
        evaluate_only=True,
    )

    assert set(evaluation) == {"train"}
    evaluation_record = json.loads((tmp_path / "evaluation" / "evaluation.json").read_text())
    assert evaluation_record["source_epoch"] is None
    assert evaluation_record["source_global_step"] == 0
    for original, current in zip(
        jax.tree_util.tree_leaves(original_params),
        jax.tree_util.tree_leaves(evaluation_model._params),
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)
    assert "Epoch 1/1" not in capsys.readouterr().out

    deferred_data = DummyDataModule()
    train(
        DummyModel(),
        deferred_data,
        specs,
        learning_rate=1e-30,
        weight_decay=0.0,
        num_epochs=3,
        heads_only=True,
        train_lora=True,
        eval_splits=("valid", "test"),
        defer_test_evaluation=True,
        prefetch_batches=0,
    )

    assert deferred_data.calls["valid"] == 3
    assert deferred_data.calls["test"] == 1


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


def test_channel_masked_double_centering_matches_sliced_channels():
    targets = jnp.square(jnp.arange(30, dtype=jnp.float32)).reshape(2, 3, 5)
    predictions = jnp.sin(targets / 7.0)
    valid_channels = jnp.asarray([True, False, True, True, False])
    mask = jnp.broadcast_to(valid_channels, targets.shape)

    masked_loss = _double_centered_correlation_loss(predictions, targets, mask)
    sliced_loss = _double_centered_correlation_loss(
        predictions[..., valid_channels], targets[..., valid_channels]
    )

    np.testing.assert_allclose(masked_loss, sliced_loss, rtol=1e-6, atol=1e-6)


def test_double_centered_correlation_loss_has_finite_gradient_without_variance():
    prediction = jnp.ones((2, 4, 3), dtype=jnp.float32)
    targets = jnp.ones((2, 4, 3), dtype=jnp.float32)

    loss, gradient = jax.value_and_grad(_double_centered_correlation_loss)(prediction, targets)

    assert loss == 0
    assert np.all(np.isfinite(gradient))
    np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


def test_row_centered_correlation_loss_removes_observation_offsets():
    targets = jnp.square(jnp.arange(24, dtype=jnp.float32)).reshape(2, 4, 3)
    observation_offset = jnp.arange(8, dtype=jnp.float32).reshape(2, 4, 1)

    loss = _row_centered_correlation_loss(targets + observation_offset, targets)

    np.testing.assert_allclose(loss, 0.0, atol=1e-6)


def test_row_centered_correlation_loss_retains_track_offsets():
    targets = jnp.square(jnp.arange(24, dtype=jnp.float32)).reshape(2, 4, 3)
    track_offset = jnp.asarray([20.0, -10.0, 5.0]).reshape(1, 1, 3)

    loss = _row_centered_correlation_loss(targets + track_offset, targets)

    assert float(loss) > 0.0


def test_row_centered_correlation_loss_ignores_masked_rows():
    targets = jnp.square(jnp.arange(18, dtype=jnp.float32)).reshape(1, 6, 3)
    predictions = targets.at[:, -2:, :].set(-1000.0)
    mask = jnp.asarray([[True, True, True, True, False, False]])

    loss = _row_centered_correlation_loss(predictions, targets, mask)

    np.testing.assert_allclose(loss, 0.0, atol=1e-6)


def test_row_centered_correlation_loss_has_finite_gradient_without_variance():
    prediction = jnp.ones((2, 4, 3), dtype=jnp.float32)
    targets = jnp.ones((2, 4, 3), dtype=jnp.float32)

    loss, gradient = jax.value_and_grad(_row_centered_correlation_loss)(prediction, targets)

    assert loss == 0
    assert np.all(np.isfinite(gradient))
    np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


def test_row_centered_correlation_loss_has_signal_for_one_gene() -> None:
    prediction = jnp.asarray([[[1.0, 4.0, 2.0]]], dtype=jnp.float32)
    targets = jnp.asarray([[[1.0, 2.0, 4.0]]], dtype=jnp.float32)

    loss, gradient = jax.value_and_grad(_row_centered_correlation_loss)(prediction, targets)

    assert float(loss) > 0
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0


def test_row_centered_correlation_loss_uses_global_pmap_batch():
    device_count = jax.local_device_count()
    targets = jnp.square(jnp.arange(device_count * 24, dtype=jnp.float32)).reshape(
        device_count, 2, 4, 3
    )
    predictions = targets.at[0, :, :, 0].set(jnp.flip(targets[0, :, :, 0], axis=1))
    expected = _row_centered_correlation_loss(
        predictions.reshape((-1, 4, 3)),
        targets.reshape((-1, 4, 3)),
    )

    @functools.partial(jax.pmap, axis_name="data")
    def distributed_loss(prediction, target):
        return _row_centered_correlation_loss(prediction, target, axis_name="data")

    actual = distributed_loss(predictions, targets)

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_double_centered_correlation_loss_uses_global_pmap_batch():
    device_count = jax.local_device_count()
    targets = jnp.square(jnp.arange(device_count * 24, dtype=jnp.float32)).reshape(
        device_count, 2, 4, 3
    )
    predictions = targets.at[0, :, :, 0].set(jnp.flip(targets[0, :, :, 0], axis=1))
    expected = _double_centered_correlation_loss(
        predictions.reshape((-1, 4, 3)),
        targets.reshape((-1, 4, 3)),
    )

    @functools.partial(jax.pmap, axis_name="data")
    def distributed_loss(prediction, target):
        return _double_centered_correlation_loss(
            prediction,
            target,
            axis_name="data",
        )

    actual = distributed_loss(predictions, targets)

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_global_pmap_correlation_gradient_matches_concatenated_batch():
    device_count = jax.local_device_count()
    features = jnp.arange(device_count * 24, dtype=jnp.float32).reshape(
        device_count, 2, 4, 3
    )
    targets = jnp.square(features + 1.0)
    baseline = jnp.sin(features / 5.0)
    direction = jnp.mod(features, 5.0)
    parameter = jnp.asarray(0.7, dtype=jnp.float32)

    def concatenated_loss(value):
        return _double_centered_correlation_loss(
            (baseline + direction * value).reshape((-1, 4, 3)),
            targets.reshape((-1, 4, 3)),
        )

    expected = jax.grad(concatenated_loss)(parameter)

    @functools.partial(jax.pmap, axis_name="data")
    def distributed_gradient(value, local_baseline, local_direction, target):
        gradient = jax.grad(
            lambda current: _double_centered_correlation_loss(
                local_baseline + local_direction * current,
                target,
                axis_name="data",
            )
        )(value)
        return jax.lax.pmean(gradient, axis_name="data")

    actual = distributed_gradient(
        jnp.broadcast_to(parameter, (device_count,)),
        baseline,
        direction,
        targets,
    )

    assert abs(expected) > 1e-3
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


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


def test_r2_stats_bins_matching_one_base_prediction_for_differential_metric():
    values = jnp.arange(8192 * 3, dtype=jnp.float32).reshape(1, 8192, 3)
    targets = jnp.sin(values / 17.0) + jnp.cos(values / 31.0)

    stats = jax.tree_util.tree_map(
        np.asarray,
        _r2_stats({"predictions_1bp": targets}, targets),
    )
    metrics = _finalize_r2_stats(stats)

    np.testing.assert_allclose(metrics["differential_pearson_r"], 1.0, rtol=1e-6)


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


def test_channel_masked_metrics_match_sliced_channels():
    targets = jnp.square(jnp.arange(30, dtype=jnp.float32)).reshape(2, 3, 5)
    predictions = jnp.sin(targets / 7.0)
    valid_channels = jnp.asarray([True, False, True, True, False])
    mask = jnp.broadcast_to(valid_channels, targets.shape)

    masked = _finalize_r2_stats(
        jax.tree_util.tree_map(np.asarray, _r2_stats(predictions, targets, mask))
    )
    sliced = _finalize_r2_stats(
        jax.tree_util.tree_map(
            np.asarray,
            _r2_stats(predictions[..., valid_channels], targets[..., valid_channels]),
        )
    )

    for metric in masked:
        np.testing.assert_allclose(masked[metric], sliced[metric], rtol=1e-6, atol=1e-6)


def test_gene_cpm_differential_metric_is_stable_at_large_offsets():
    row = jnp.arange(12, dtype=jnp.float32).reshape(1, 12, 1)
    track = jnp.arange(5, dtype=jnp.float32).reshape(1, 1, 5)
    targets = 500_000.0 + 10_000.0 * row + 20_000.0 * track + 500.0 * row * track
    predictions = targets + 200.0 * jnp.sin(row + track)
    valid = jnp.ones((1, 12), dtype=bool)

    metrics = _finalize_r2_stats(
        jax.tree_util.tree_map(np.asarray, _r2_stats(predictions, targets, valid))
    )
    def double_center(values):
        return (
            values
            - values.mean(1, keepdims=True)
            - values.mean(2, keepdims=True)
            + values.mean((1, 2), keepdims=True)
        )

    expected = np.corrcoef(
        np.asarray(double_center(predictions)).ravel(),
        np.asarray(double_center(targets)).ravel(),
    )[0, 1]

    assert -1.0 <= metrics["differential_pearson_r"] <= 1.0
    np.testing.assert_allclose(metrics["differential_pearson_r"], expected, rtol=2e-4)


def test_gene_expression_prediction_selects_strand_and_exon_fraction():
    prediction = {
        "predictions_128bp": jnp.asarray([[[8.0, 80.0, 4.0, 40.0], [16.0, 160.0, 2.0, 20.0]]])
    }
    batch = {
        "gene_weights_rna": jnp.asarray([[[0.5, 0.25], [1.0, 0.5]]]),
        "gene_targets_rna": jnp.zeros((1, 2, 2)),
        "gene_strands_rna": jnp.asarray([[0, 1]]),
    }

    result = np.asarray(_gene_expression_prediction(prediction, batch, "rna"))

    np.testing.assert_allclose(result[0, 0], [20.0, 4.0])
    np.testing.assert_allclose(result[0, 1], [100.0, 20.0])


def test_gene_expression_prediction_supports_unstranded_channels():
    prediction = {
        "predictions_128bp": jnp.asarray([[[8.0, 4.0], [16.0, 2.0]]])
    }
    batch = {
        "gene_weights_rna": jnp.asarray([[[0.5, 0.25], [1.0, 0.5]]]),
        "gene_targets_rna": jnp.zeros((1, 2, 2)),
        "gene_strands_rna": jnp.asarray([[0, 1]]),
    }

    result = np.asarray(_gene_expression_prediction(prediction, batch, "rna"))

    np.testing.assert_allclose(result, [[[20.0, 4.0], [10.0, 2.0]]])
