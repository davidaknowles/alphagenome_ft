"""Training utilities for fine-tuning AlphaGenome models."""

from __future__ import annotations

import functools
import json
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
import orbax.checkpoint as ocp
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model

from alphagenome_ft.lora import ADAPTER_LEAF_NAMES
from alphagenome_ft import parameter_utils
from alphagenome_ft.custom_model import CustomAlphaGenomeModel
from alphagenome_ft.finetune.config import HeadSpec
from alphagenome_ft.finetune.data import BigWigDataModule, prepare_batch
from alphagenome_ft.finetune.target_transforms import load_target_transform
from alphagenome_ft.optimizer_utils import create_optimizer


def register_predefined_heads(head_specs: Sequence[HeadSpec]) -> None:
    """Register predefined heads from parsed head specs."""
    from alphagenome_ft import register_predefined_head

    for spec in head_specs:
        if spec.source != "predefined":
            continue
        if spec.config is None:
            raise ValueError(f'Predefined head "{spec.head_id}" missing config.')
        register_predefined_head(
            spec.head_id,
            spec.config,
            metadata=spec.metadata,
        )


def _keypath_to_str(path_tuple: tuple) -> str:
    """Convert a JAX parameter key-path tuple to a slash-delimited string."""
    parts = []
    for key in path_tuple:
        if isinstance(key, parameter_utils.DictKey):
            parts.append(str(key.key))
        elif isinstance(key, parameter_utils.GetAttrKey):
            parts.append(str(key.name))
        elif isinstance(key, parameter_utils.SequenceKey):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return "/".join(parts)


def _is_trainable_head_path(path_str: str, trainable_heads: set[str]) -> bool:
    """Return True if a parameter path belongs to any requested trainable head."""
    for head_name in trainable_heads:
        if f"/head/{head_name}/" in path_str or path_str.startswith(f"head/{head_name}/"):
            return True
    return False


def _is_lora_path(path_str: str) -> bool:
    return path_str.split("/")[-1] in ADAPTER_LEAF_NAMES


def _label_params_for_heads(
    params,
    trainable_heads: Sequence[str],
    *,
    train_lora: bool = False,
):
    """Label model parameters as trainable params vs frozen params."""
    head_set = {str(name) for name in trainable_heads}

    def label_fn(path, _value):
        path_str = _keypath_to_str(path)
        if _is_trainable_head_path(path_str, head_set):
            return "train"
        if train_lora and _is_lora_path(path_str):
            return "train"
        return "frozen"

    return jax.tree_util.tree_map_with_path(label_fn, params)


def create_optimizer(
    params,
    trainable_head_names: Sequence[str],
    learning_rate: float,
    weight_decay: float,
    heads_only: bool,
    train_lora: bool = False,
):
    """Create optimizer for full finetuning or masked head/LoRA finetuning."""
    if heads_only:
        head_set = {str(name) for name in trainable_head_names}
        head_paths = parameter_utils.get_head_parameter_paths(params)
        matched_paths = [path for path in head_paths if _is_trainable_head_path(path, head_set)]
        if not matched_paths:
            sample_paths = ", ".join(head_paths[:5]) if head_paths else "<none>"
            raise ValueError(
                "No trainable head parameters matched --heads-only filter. "
                f"Names tried: {sorted(head_set)}. "
                f"Head parameter sample: {sample_paths}"
            )
        if train_lora:
            lora_paths = [
                path for path in parameter_utils.get_parameter_paths(params) if _is_lora_path(path)
            ]
            if not lora_paths:
                raise ValueError(
                    "train_lora=True was requested, but no LoRA/LoCon adapter parameters exist."
                )
        param_labels = _label_params_for_heads(
            params,
            trainable_head_names,
            train_lora=train_lora,
        )
        return optax.multi_transform(
            {
                "train": optax.adamw(learning_rate, weight_decay=weight_decay),
                "frozen": optax.set_to_zero(),
            },
            param_labels,
        )
    return optax.adamw(learning_rate, weight_decay=weight_decay)


def _replicate_tree(tree, devices):
    """Replicate a pytree across local devices for pmap."""
    mesh = Mesh(np.array(devices), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(jnp.stack([value] * len(devices)), sharding),
        tree,
    )


def _unreplicate_tree(tree):
    """Extract the first local replica from a replicated pytree."""
    return jax.tree_util.tree_map(
        lambda value: (
            np.asarray(value.addressable_shards[0].data).squeeze(0)
            if hasattr(value, "addressable_shards")
            else value[0]
        ),
        tree,
    )


def _shard_batch(batch: Mapping[str, jax.Array], num_devices: int):
    """Reshape a batch from [global_batch, ...] to [num_devices, per_device, ...]."""

    def shard_array(value: jax.Array) -> jax.Array:
        if value.shape[0] % num_devices != 0:
            raise ValueError(
                f"Batch size of {value.shape[0]} is not divisible by num_devices={num_devices}."
                f"Use a global batch size divisible by the number of local devices."
            )
        per_device_batch = value.shape[0] // num_devices
        return value.reshape((num_devices, per_device_batch, *value.shape[1:]))

    return {name: shard_array(value) for name, value in batch.items()}


def _json_ready(value: Any):
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_json_ready(payload), sort_keys=True) + "\n")


def _select_prediction_for_targets(prediction, targets):
    """Return the prediction array that matches 1 bp BigWig targets."""
    if hasattr(prediction, "shape"):
        return prediction
    if not isinstance(prediction, Mapping):
        raise TypeError(f"Unsupported prediction type for R2 metrics: {type(prediction)!r}")

    preferred_keys = (
        "predictions_1bp",
        "predictions",
        "scaled_predictions_1bp",
    )
    for key in preferred_keys:
        value = prediction.get(key)
        if hasattr(value, "shape") and value.shape == targets.shape:
            return value

    for key, value in prediction.items():
        if (
            str(key).startswith("predictions_")
            and hasattr(value, "shape")
            and value.shape == targets.shape
        ):
            return value
    for key, value in prediction.items():
        if (
            str(key).startswith("scaled_predictions_")
            and hasattr(value, "shape")
            and value.shape == targets.shape
        ):
            return value

    shapes = {str(key): getattr(value, "shape", None) for key, value in prediction.items()}
    raise ValueError(
        "Could not find a prediction array matching target shape "
        f"{targets.shape}; available prediction shapes: {shapes}"
    )


def _maybe_bin_128bp_jax(value):
    if len(value.shape) != 3:
        return value
    loci = value.shape[1]
    if loci >= 8192 and loci % 128 == 0:
        return jnp.mean(
            value.reshape((value.shape[0], loci // 128, 128, value.shape[2])),
            axis=2,
        )
    return value


def _align_prediction_and_targets(prediction, targets, observation_mask=None):
    try:
        return _select_prediction_for_targets(prediction, targets), targets
    except ValueError:
        if not isinstance(prediction, Mapping) or observation_mask is not None:
            raise

    candidates = []
    for key, value in prediction.items():
        if (
            str(key).startswith("predictions_")
            and hasattr(value, "shape")
            and len(value.shape) == len(targets.shape) == 3
            and value.shape[0] == targets.shape[0]
            and value.shape[-1] == targets.shape[-1]
            and value.shape[1] < targets.shape[1]
            and targets.shape[1] % value.shape[1] == 0
        ):
            candidates.append(value)
    if not candidates:
        return _select_prediction_for_targets(prediction, targets), targets

    selected = max(candidates, key=lambda value: value.shape[1])
    width = targets.shape[1] // selected.shape[1]
    pooled_targets = targets.reshape(
        targets.shape[0], selected.shape[1], width, targets.shape[-1]
    ).sum(axis=2, dtype=jnp.float32)
    return selected, pooled_targets


def _r2_stats(prediction, targets, observation_mask=None):
    prediction, targets = _align_prediction_and_targets(prediction, targets, observation_mask)
    prediction = prediction.astype(jnp.float32)
    targets = targets.astype(jnp.float32)
    residual = prediction - targets

    if observation_mask is None:
        mask = jnp.ones(targets.shape[:-1], dtype=jnp.float32)
    else:
        mask = observation_mask.astype(jnp.float32)
        if mask.shape != targets.shape[:-1]:
            raise ValueError(
                f"Observation mask shape {mask.shape} does not match {targets.shape[:-1]}."
            )
    mask_channels = mask[..., None]

    count = jnp.sum(mask) * targets.shape[-1]
    sum_y = jnp.sum(targets * mask_channels)
    sum_y2 = jnp.sum(jnp.square(targets) * mask_channels)
    sse = jnp.sum(jnp.square(residual) * mask_channels)

    loci_count = jnp.sum(mask)
    sum_y_by_track = jnp.sum(targets * mask_channels, axis=(0, 1))
    sum_y2_by_track = jnp.sum(jnp.square(targets) * mask_channels, axis=(0, 1))
    sse_by_track = jnp.sum(jnp.square(residual) * mask_channels, axis=(0, 1))
    count_by_track = jnp.ones_like(sum_y_by_track, dtype=jnp.float32) * loci_count

    prediction_bins = _maybe_bin_128bp_jax(prediction)
    target_bins = _maybe_bin_128bp_jax(targets)
    pred_matrix = prediction_bins.reshape((-1, prediction_bins.shape[-1]))
    target_matrix = target_bins.reshape((-1, target_bins.shape[-1]))
    if observation_mask is None:
        differential_mask = jnp.ones((pred_matrix.shape[0],), dtype=jnp.float32)
    else:
        differential_mask = mask.reshape((-1,))
    differential_mask_channels = differential_mask[:, None]
    pred_matrix = pred_matrix * differential_mask_channels
    target_matrix = target_matrix * differential_mask_channels
    pred_row_sum = jnp.sum(pred_matrix, axis=-1)
    target_row_sum = jnp.sum(target_matrix, axis=-1)
    pred_track_sum = jnp.sum(pred_matrix, axis=0)
    target_track_sum = jnp.sum(target_matrix, axis=0)
    differential_count = jnp.sum(differential_mask)
    differential_pred_sum = jnp.sum(pred_matrix)
    differential_target_sum = jnp.sum(target_matrix)

    target_mean_by_locus = jnp.mean(targets, axis=-1, keepdims=True)
    sst_by_locus = jnp.sum(jnp.square(targets - target_mean_by_locus), axis=-1)
    sse_by_locus = jnp.sum(jnp.square(residual), axis=-1)
    valid_locus = (sst_by_locus > 0) & (mask > 0)
    r2_by_locus = 1.0 - (sse_by_locus / jnp.maximum(sst_by_locus, 1e-8))
    r2_cell_type_sum = jnp.sum(jnp.where(valid_locus, r2_by_locus, 0.0))
    r2_cell_type_count = jnp.sum(valid_locus.astype(jnp.float32))

    return {
        "count": count,
        "sum_y": sum_y,
        "sum_y2": sum_y2,
        "sse": sse,
        "count_by_track": count_by_track,
        "sum_y_by_track": sum_y_by_track,
        "sum_y2_by_track": sum_y2_by_track,
        "sse_by_track": sse_by_track,
        "differential_count": differential_count,
        "differential_pred_sum": differential_pred_sum,
        "differential_target_sum": differential_target_sum,
        "differential_pred2_sum": jnp.sum(jnp.square(pred_matrix)),
        "differential_target2_sum": jnp.sum(jnp.square(target_matrix)),
        "differential_pred_target_sum": jnp.sum(pred_matrix * target_matrix),
        "differential_pred_row2_sum": jnp.sum(jnp.square(pred_row_sum)),
        "differential_target_row2_sum": jnp.sum(jnp.square(target_row_sum)),
        "differential_pred_target_row_sum": jnp.sum(pred_row_sum * target_row_sum),
        "differential_pred_track_sum": pred_track_sum,
        "differential_target_track_sum": target_track_sum,
        "r2_cell_type_sum": r2_cell_type_sum,
        "r2_cell_type_count": r2_cell_type_count,
    }


def _gene_expression_prediction(prediction, batch, head_name: str):
    """Sum 128 bp predictions over annotated exons on each gene's strand."""
    if not isinstance(prediction, Mapping) or "predictions_128bp" not in prediction:
        raise ValueError(f"Head {head_name} requires predictions_128bp for gene supervision.")
    prediction_128bp = prediction["predictions_128bp"].astype(jnp.float32)
    weights = batch[f"gene_weights_{head_name}"].astype(jnp.float32)
    if prediction_128bp.shape[-1] % 2:
        raise ValueError(f"Head {head_name} must have paired strand channels.")
    positive = jnp.einsum("bsg,bsc->bgc", weights, prediction_128bp[..., 0::2])
    negative = jnp.einsum("bsg,bsc->bgc", weights, prediction_128bp[..., 1::2])
    strands = batch[f"gene_strands_{head_name}"][..., None]
    return jnp.where(strands == 0, positive, negative)


def _gene_log_mse(prediction, targets, valid):
    residual = jnp.log1p(jnp.maximum(prediction, 0.0)) - jnp.log1p(targets.astype(jnp.float32))
    mask = valid.astype(jnp.float32)[..., None]
    return jnp.sum(jnp.square(residual) * mask) / jnp.maximum(
        jnp.sum(mask) * targets.shape[-1], 1.0
    )


def _double_centered_correlation_loss(prediction, targets, observation_mask=None):
    """Return one minus signed Pearson correlation after two-axis centering."""
    prediction, targets = _align_prediction_and_targets(prediction, targets, observation_mask)
    prediction = prediction.astype(jnp.float32)
    targets = targets.astype(jnp.float32)
    if observation_mask is None:
        prediction = _maybe_bin_128bp_jax(prediction)
        targets = _maybe_bin_128bp_jax(targets)
        mask = jnp.ones(prediction.shape[:-1], dtype=jnp.float32)
    else:
        mask = observation_mask.astype(jnp.float32)
        if mask.shape != targets.shape[:-1]:
            raise ValueError(
                f"Observation mask shape {mask.shape} does not match {targets.shape[:-1]}."
            )

    pred_matrix = prediction.reshape((-1, prediction.shape[-1]))
    target_matrix = targets.reshape((-1, targets.shape[-1]))
    row_mask = mask.reshape((-1, 1))
    count = jnp.sum(row_mask)
    num_tracks = prediction.shape[-1]

    def center(values):
        values = values * row_mask
        track_mean = jnp.sum(values, axis=0, keepdims=True) / jnp.maximum(count, 1.0)
        row_mean = jnp.mean(values, axis=1, keepdims=True)
        grand_mean = jnp.sum(values) / jnp.maximum(count * num_tracks, 1.0)
        return (values - track_mean - row_mean + grand_mean) * row_mask

    pred_centered = center(pred_matrix)
    target_centered = center(target_matrix)
    covariance = jnp.sum(pred_centered * target_centered)
    pred_sum_squares = jnp.sum(jnp.square(pred_centered))
    target_sum_squares = jnp.sum(jnp.square(target_centered))
    denominator = jnp.sqrt(jnp.maximum(pred_sum_squares * target_sum_squares, 1e-16))
    correlation = covariance / denominator
    has_variance = (pred_sum_squares > 1e-8) & (target_sum_squares > 1e-8)
    return jnp.where((count > 0) & has_variance, 1.0 - correlation, 0.0)


def _weighted_head_loss_sum(head_losses, head_specs_by_name):
    """Sum head objectives using configured modality weights."""
    total_loss = 0.0
    for head_name, head_loss in head_losses.items():
        total_loss = total_loss + head_specs_by_name[head_name].loss_weight * head_loss
    return total_loss


def _save_optimizer_state(path: Path, opt_state) -> None:
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(str(path), opt_state, force=True)
    checkpointer.wait_until_finished()


def _restore_optimizer_state(path: Path, target):
    checkpointer = ocp.StandardCheckpointer()
    return checkpointer.restore(str(path), target=target)


def _finalize_r2_stats(stats: Mapping[str, np.ndarray | float]) -> dict[str, float]:
    count = float(np.asarray(stats["count"]))
    sst = float(np.asarray(stats["sum_y2"]) - np.asarray(stats["sum_y"]) ** 2 / max(count, 1.0))
    sse = float(np.asarray(stats["sse"]))
    r2_global = float("nan") if sst <= 0 else 1.0 - sse / sst

    count_by_track = np.asarray(stats["count_by_track"], dtype=np.float64)
    sum_y_by_track = np.asarray(stats["sum_y_by_track"], dtype=np.float64)
    sum_y2_by_track = np.asarray(stats["sum_y2_by_track"], dtype=np.float64)
    sse_by_track = np.asarray(stats["sse_by_track"], dtype=np.float64)
    sst_by_track = sum_y2_by_track - np.square(sum_y_by_track) / np.maximum(count_by_track, 1.0)
    valid_tracks = sst_by_track > 0
    if np.any(valid_tracks):
        r2_over_loci = float(np.mean(1.0 - sse_by_track[valid_tracks] / sst_by_track[valid_tracks]))
    else:
        r2_over_loci = float("nan")

    r2_cell_type_count = float(np.asarray(stats["r2_cell_type_count"]))
    if r2_cell_type_count > 0:
        r2_over_cell_types = float(np.asarray(stats["r2_cell_type_sum"]) / r2_cell_type_count)
    else:
        r2_over_cell_types = float("nan")
    differential_count = float(np.asarray(stats["differential_count"]))
    pred_sum = float(np.asarray(stats["differential_pred_sum"]))
    target_sum = float(np.asarray(stats["differential_target_sum"]))
    pred_track_sum = np.asarray(stats["differential_pred_track_sum"], dtype=np.float64)
    target_track_sum = np.asarray(stats["differential_target_track_sum"], dtype=np.float64)
    differential_tracks = float(pred_track_sum.shape[0])
    pred_ss = (
        float(np.asarray(stats["differential_pred2_sum"]))
        - float(np.asarray(stats["differential_pred_row2_sum"])) / differential_tracks
        - float(np.sum(np.square(pred_track_sum))) / differential_count
        + pred_sum * pred_sum / (differential_count * differential_tracks)
    )
    target_ss = (
        float(np.asarray(stats["differential_target2_sum"]))
        - float(np.asarray(stats["differential_target_row2_sum"])) / differential_tracks
        - float(np.sum(np.square(target_track_sum))) / differential_count
        + target_sum * target_sum / (differential_count * differential_tracks)
    )
    pred_target_cov = (
        float(np.asarray(stats["differential_pred_target_sum"]))
        - float(np.asarray(stats["differential_pred_target_row_sum"])) / differential_tracks
        - float(np.sum(pred_track_sum * target_track_sum)) / differential_count
        + pred_sum * target_sum / (differential_count * differential_tracks)
    )
    if pred_ss <= 0 or target_ss <= 0:
        differential_pearson_r = float("nan")
    else:
        differential_pearson_r = float(pred_target_cov / np.sqrt(pred_ss * target_ss))
    double_centered_r2 = differential_pearson_r * differential_pearson_r

    return {
        "r2_global": r2_global,
        "r2_over_loci": r2_over_loci,
        "r2_over_cell_types": r2_over_cell_types,
        "differential_pearson_r": differential_pearson_r,
        "double_centered_r2": double_centered_r2,
    }


def _add_stats(total: dict[str, Any] | None, update: Mapping[str, Any]) -> dict[str, Any]:
    update_np = {key: np.asarray(value) for key, value in update.items()}
    if total is None:
        return {key: np.array(value) for key, value in update_np.items()}
    for key, value in update_np.items():
        total[key] = total[key] + value
    return total


def _add_device_stats(total, update):
    if total is None:
        return update
    return jax.tree_util.tree_map(lambda left, right: left + right, total, update)


def _prefetch_iterable(iterable: Iterable[Any], buffer_size: int) -> Iterator[Any]:
    if buffer_size <= 0:
        yield from iterable
        return

    item_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=buffer_size)
    stop_event = threading.Event()

    def put_item(item: tuple[str, Any]) -> bool:
        while not stop_event.is_set():
            try:
                item_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def worker() -> None:
        try:
            for item in iterable:
                if stop_event.is_set():
                    break
                if not put_item(("item", item)):
                    break
        except BaseException as exc:
            put_item(("error", exc))
        finally:
            put_item(("done", None))

    thread = threading.Thread(target=worker, name="alphagenome-batch-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            kind, payload = item_queue.get()
            if kind == "item":
                yield payload
            elif kind == "error":
                raise payload
            else:
                break
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


def _prefetch_transformed_iterable(
    iterable: Iterable[Any],
    buffer_size: int,
    transform,
) -> Iterator[Any]:
    if buffer_size <= 0:
        for item in iterable:
            yield transform(item)
        return

    item_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=buffer_size)
    stop_event = threading.Event()

    def put_item(item: tuple[str, Any]) -> bool:
        while not stop_event.is_set():
            try:
                item_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def worker() -> None:
        try:
            for item in iterable:
                if stop_event.is_set():
                    break
                if not put_item(("item", transform(item))):
                    break
        except BaseException as exc:
            put_item(("error", exc))
        finally:
            put_item(("done", None))

    thread = threading.Thread(
        target=worker,
        name="alphagenome-device-batch-prefetch",
        daemon=True,
    )
    thread.start()
    try:
        while True:
            kind, payload = item_queue.get()
            if kind == "item":
                yield payload
            elif kind == "error":
                raise payload
            else:
                break
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


def _format_host_timing_summary(title: str, stats: Mapping[str, float], count: int) -> str:
    parts = []
    if count <= 0:
        return f"  {title}: no samples"
    for label, elapsed in stats.items():
        if elapsed <= 0:
            continue
        parts.append(f"{label}={elapsed:.3f}s ({elapsed / count:.4f}s/step)")
    if not parts:
        return f"  {title}: no samples"
    return f"  {title}: " + "; ".join(parts)


def _flatten_valid_metrics(
    metrics_by_head: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Flatten per-head validation metrics using the checkpoint-selection keys."""
    flattened: dict[str, float] = {}
    for head, values in metrics_by_head.items():
        flattened[head] = values["loss"]
        for metric_name, metric_value in values.items():
            flattened.setdefault(metric_name, metric_value)
            flattened[f"{head}/{metric_name}"] = metric_value
    if metrics_by_head:
        metric_names = set.intersection(*(set(values) for values in metrics_by_head.values()))
        for metric_name in metric_names:
            values = [head_values[metric_name] for head_values in metrics_by_head.values()]
            finite_values = [value for value in values if math.isfinite(value)]
            if finite_values:
                flattened[f"mean/{metric_name}"] = float(sum(finite_values) / len(finite_values))
    return flattened


def train(
    model: CustomAlphaGenomeModel,
    data_module: BigWigDataModule,
    head_specs: Sequence[HeadSpec],
    *,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    seed: int = 42,
    max_train_steps: int | None = None,
    heads_only: bool = False,
    train_lora: bool = False,
    checkpoint_dir: Path | None = None,
    organism: str = "HOMO_SAPIENS",
    best_metric: str = "valid_loss",
    best_metric_mode: str = "min",
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    verbose: bool = False,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: Sequence[str] | None = None,
    wandb_job_type: str | None = None,
    wandb_mode: str | None = None,
    wandb_config: dict | None = None,
    num_devices: int = 1,
    eval_splits: Sequence[str] = ("valid",),
    progress_interval: int = 50,
    prefetch_batches: int = 2,
    profile_host_timing: bool = False,
    start_epoch: int = 1,
    initial_global_step: int = 0,
    initial_optimizer_state_path: Path | None = None,
) -> None:
    """Run fine-tuning with pmapped train/eval steps.

    Args:
        model: Initialized AlphaGenome model wrapper to fine-tune.
        data_module: Batch provider with train/valid intervals and BigWig targets.
        head_specs: Head definitions used to build losses and optimizer filters.
        learning_rate: Base AdamW learning rate.
        weight_decay: AdamW weight decay.
        num_epochs: Maximum number of epochs to run.
        seed: Base RNG seed used for per-epoch training shuffles.
        max_train_steps: Optional global cap on optimizer updates across all epochs.
        heads_only: If True, freeze backbone and optimize selected heads only.
        train_lora: If True with ``heads_only=True``, also optimize LoRA adapter
            leaves named ``lora_a`` or ``lora_b`` outside selected heads.
        checkpoint_dir: Optional output directory for ``best``/``last`` checkpoints.
        organism: Organism enum name used for model organism indexing.
        best_metric: Metric name used for best-checkpoint and early-stopping tracking.
        best_metric_mode: Improvement direction for ``best_metric`` (``min`` or ``max``).
        early_stopping_patience: Stop after this many non-improving epochs (0 disables).
        early_stopping_min_delta: Minimum metric change required to count as improvement.
        verbose: If True, print per-step progress and extra diagnostics.
        use_wandb: If True, log metrics to Weights & Biases.
        wandb_project: Optional W&B project name override.
        wandb_entity: Optional W&B entity/team override.
        wandb_run_name: Optional W&B run-name override.
        wandb_group: Optional W&B group for related runs.
        wandb_tags: Optional W&B tags.
        wandb_job_type: Optional W&B job type.
        wandb_mode: Optional W&B mode, such as ``online`` or ``offline``.
        wandb_config: Optional extra config keys to merge into W&B config.
        num_devices: Number of local devices to use. Defaults to single-device.
        eval_splits: Data splits to evaluate after each epoch. Supported values
            are any split present in the data module, typically ``train``,
            ``valid``, and ``test``.
        progress_interval: Synchronize and report per-step loss every this many
            steps when verbose or W&B step logging is enabled. Larger values
            improve overlap between host-side data loading and GPU execution.
        prefetch_batches: Number of host-prepared batches to keep queued in a
            background thread. Set to 0 to disable prefetching.
        profile_host_timing: If True, print wall-clock timing buckets for host-side
            work each epoch.
        start_epoch: One-indexed epoch at which to continue training. Values above
            one require prior metric history in ``checkpoint_dir``.
        initial_global_step: Number of optimizer updates completed before this call.
        initial_optimizer_state_path: Optional optimizer checkpoint matching the
            resumed model parameters. Existing parameter-only checkpoints omit it.

    Notes:
        Total planned steps are computed before training from train-set size and
        batch settings as ``steps_per_epoch * num_epochs`` (or capped by
        ``max_train_steps`` when provided). Progress is reported with a global
        counter in ``current/total`` format.

        Multi-GPU training on a single node is supported by passing a non-zero ``num_devices``
        with a global ``batch_size`` divisible by ``num_devices``. For multi-GPU training,
        the code requires ``drop_last=True`` to ensure all batches are evenly divisible
        across devices.

        The multi-GPU implementation is a distributed data-parallel (DDP) style approach
        using JAX's ``pmap``. Model parameters and optimizer state are replicated across
        devices, and each device processes a shard of each batch. Gradients and metrics
        are averaged across devices with ``lax.pmean`` to keep them in sync.
    """
    train_intervals = list(data_module._intervals.get("train", ()))
    num_train_examples = len(train_intervals)
    if num_train_examples == 0:
        raise ValueError("No train intervals available for training.")

    if data_module._drop_last:
        steps_per_epoch = num_train_examples // data_module._batch_size
    else:
        steps_per_epoch = math.ceil(num_train_examples / data_module._batch_size)
    if steps_per_epoch == 0:
        raise ValueError(
            "Computed zero training steps per epoch. Check batch size, drop_last, and train intervals."
        )

    planned_steps = steps_per_epoch * num_epochs
    total_train_steps = (
        min(planned_steps, max_train_steps) if max_train_steps is not None else planned_steps
    )
    step_width = len(str(steps_per_epoch))

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if num_devices < 1:
        raise ValueError(f"num_devices must be at least 1, got {num_devices}.")
    if progress_interval < 1:
        raise ValueError(f"progress_interval must be at least 1, got {progress_interval}.")
    if prefetch_batches < 0:
        raise ValueError(f"prefetch_batches must be non-negative, got {prefetch_batches}.")
    if start_epoch < 1 or start_epoch > num_epochs:
        raise ValueError(
            f"start_epoch must be between 1 and num_epochs={num_epochs}, got {start_epoch}."
        )
    if initial_global_step < 0:
        raise ValueError("initial_global_step must be non-negative.")

    available_devices = jax.local_devices()
    if num_devices > len(available_devices):
        raise ValueError(
            f"Requested num_devices={num_devices}, but only {len(available_devices)} local "
            f"device(s) are available."
        )
    devices = available_devices[:num_devices]

    if use_wandb:
        import wandb

        wb_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "batch_size": data_module._batch_size,
            "steps_per_epoch": steps_per_epoch,
            "total_train_steps": total_train_steps,
            "heads_only": heads_only,
            "train_lora": train_lora,
            "organism": organism,
            "num_devices": num_devices,
            "best_metric": best_metric,
            "best_metric_mode": best_metric_mode,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
            "eval_splits": list(eval_splits),
            "progress_interval": progress_interval,
            "prefetch_batches": prefetch_batches,
            **(wandb_config or {}),
        }
        wandb.init(
            project=wandb_project or "alphagenome-ft",
            entity=wandb_entity,
            name=wandb_run_name,
            group=wandb_group,
            tags=list(wandb_tags or ()),
            job_type=wandb_job_type,
            mode=wandb_mode,
            config=wb_config,
        )

    head_names = [spec.head_id for spec in head_specs]
    head_specs_by_name = {spec.head_id: spec for spec in head_specs}
    if num_devices > 1 and not data_module._drop_last:
        raise ValueError(
            "Single-host multi-GPU training currently requires drop_last=True so every "
            "batch can be sharded evenly across devices."
        )
    if heads_only:
        model.freeze_backbone()

    optimizer = create_optimizer(
        model._params,
        trainable_head_names=head_names,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        heads_only=heads_only,
        train_lora=train_lora,
    )
    opt_state = optimizer.init(model._params)
    if initial_optimizer_state_path is not None:
        initial_optimizer_state_path = Path(initial_optimizer_state_path).expanduser().resolve()
        if not initial_optimizer_state_path.exists():
            raise FileNotFoundError(
                f"Optimizer state checkpoint not found: {initial_optimizer_state_path}"
            )
        opt_state = _restore_optimizer_state(initial_optimizer_state_path, opt_state)

    organism_enum = getattr(ag_dna_model.Organism, organism)
    organism_index_value = research_dna_model.convert_to_organism_index(organism_enum)
    strand_reindexing = model._metadata[organism_enum].strand_reindexing

    loss_fns = {name: model.create_loss_fn_for_head(name) for name in head_names}
    target_transforms = {
        spec.head_id: load_target_transform(spec.target_transform_path)
        for spec in head_specs
        if spec.target_transform_path is not None
    }

    def inverse_prediction_for_metrics(head_name, prediction):
        transform = target_transforms.get(head_name)
        if transform is None:
            return prediction
        return transform.inverse_jax(prediction["predictions_1bp"])

    @functools.partial(jax.pmap, axis_name="data")
    def train_step(params, state, current_opt_state, batch):
        def loss_fn(current_params):
            predictions = model._predict(
                current_params,
                state,
                batch["sequences"],
                batch["organism_index"],
                requested_outputs=head_names,
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=batch["strand_reindexing"],
            )
            head_losses = {}
            head_stats = {}
            for head_name in head_names:
                spec = head_specs_by_name[head_name]
                uses_coverage = spec.gene_supervision_path is None or spec.coverage_loss_weight > 0
                if uses_coverage:
                    targets = batch[f"targets_{head_name}"]
                    transform = target_transforms.get(head_name)
                    if transform is not None:
                        targets = transform.forward_jax(targets)
                    head_loss = loss_fns[head_name](
                        predictions[head_name],
                        {
                            "targets": targets,
                            "organism_index": batch["organism_index"],
                        },
                    )["loss"]
                else:
                    head_loss = jnp.asarray(0.0, dtype=jnp.float32)
                if spec.gene_supervision_path is not None:
                    gene_prediction = _gene_expression_prediction(
                        predictions[head_name], batch, head_name
                    )
                    gene_targets = batch[f"gene_targets_{head_name}"]
                    gene_valid = batch[f"gene_valid_{head_name}"]
                    gene_loss = _gene_log_mse(gene_prediction, gene_targets, gene_valid)
                    head_loss = (
                        spec.coverage_loss_weight * head_loss + spec.gene_loss_weight * gene_loss
                    )
                    correlation_prediction = gene_prediction
                    correlation_targets = gene_targets
                    correlation_mask = gene_valid
                else:
                    correlation_prediction = predictions[head_name]
                    correlation_targets = targets
                    correlation_mask = None
                if spec.double_centered_correlation_loss_weight > 0:
                    head_loss = head_loss + (
                        spec.double_centered_correlation_loss_weight
                        * _double_centered_correlation_loss(
                            correlation_prediction,
                            correlation_targets,
                            correlation_mask,
                        )
                    )
                head_losses[head_name] = head_loss
                if spec.gene_supervision_path is not None:
                    head_stats[head_name] = _r2_stats(gene_prediction, gene_targets, gene_valid)
                else:
                    head_stats[head_name] = _r2_stats(predictions[head_name], targets)
            total_loss = _weighted_head_loss_sum(head_losses, head_specs_by_name)
            return total_loss, (head_losses, head_stats)

        (loss_value, (head_losses, head_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )
        loss_value = jax.lax.pmean(loss_value, axis_name="data")
        grads = jax.lax.pmean(grads, axis_name="data")
        head_losses = jax.tree_util.tree_map(
            lambda head_loss: jax.lax.pmean(head_loss, axis_name="data"),
            head_losses,
        )
        head_stats = jax.tree_util.tree_map(
            lambda value: jax.lax.psum(value, axis_name="data"),
            head_stats,
        )
        updates, new_opt_state = optimizer.update(grads, current_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_value, head_losses, head_stats

    @functools.partial(jax.pmap, axis_name="data")
    def eval_step(params, state, batch):
        predictions = model._predict(
            params,
            state,
            batch["sequences"],
            batch["organism_index"],
            requested_outputs=head_names,
            negative_strand_mask=batch["negative_strand_mask"],
            strand_reindexing=batch["strand_reindexing"],
        )
        head_losses = {}
        head_stats = {}
        for head_name in head_names:
            spec = head_specs_by_name[head_name]
            uses_coverage = spec.gene_supervision_path is None or spec.coverage_loss_weight > 0
            if uses_coverage:
                raw_targets = batch[f"targets_{head_name}"]
                transform = target_transforms.get(head_name)
                loss_targets = (
                    transform.forward_jax(raw_targets) if transform is not None else raw_targets
                )
                head_loss = loss_fns[head_name](
                    predictions[head_name],
                    {
                        "targets": loss_targets,
                        "organism_index": batch["organism_index"],
                    },
                )["loss"]
            else:
                head_loss = jnp.asarray(0.0, dtype=jnp.float32)
            if spec.gene_supervision_path is not None:
                gene_prediction = _gene_expression_prediction(
                    predictions[head_name], batch, head_name
                )
                gene_targets = batch[f"gene_targets_{head_name}"]
                gene_valid = batch[f"gene_valid_{head_name}"]
                head_loss = (
                    spec.coverage_loss_weight * head_loss
                    + spec.gene_loss_weight
                    * _gene_log_mse(gene_prediction, gene_targets, gene_valid)
                )
                correlation_prediction = gene_prediction
                correlation_targets = gene_targets
                correlation_mask = gene_valid
                head_stats[head_name] = _r2_stats(gene_prediction, gene_targets, gene_valid)
            else:
                correlation_prediction = predictions[head_name]
                correlation_targets = loss_targets
                correlation_mask = None
                head_stats[head_name] = _r2_stats(
                    inverse_prediction_for_metrics(head_name, predictions[head_name]),
                    raw_targets,
                )
            if spec.double_centered_correlation_loss_weight > 0:
                head_loss = head_loss + (
                    spec.double_centered_correlation_loss_weight
                    * _double_centered_correlation_loss(
                        correlation_prediction,
                        correlation_targets,
                        correlation_mask,
                    )
                )
            head_losses[head_name] = head_loss
        head_losses = jax.tree_util.tree_map(
            lambda loss_value: jax.lax.pmean(loss_value, axis_name="data"),
            head_losses,
        )
        head_stats = jax.tree_util.tree_map(
            lambda value: jax.lax.psum(value, axis_name="data"),
            head_stats,
        )
        return head_losses, head_stats

    if verbose:
        print("JIT-compiling step functions (first call will be slow)...")

    def aggregate_valid_loss(metrics: Mapping[str, float]) -> float | None:
        losses = [metrics[head_name] for head_name in head_names if head_name in metrics]
        return float(sum(losses)) if losses else None

    def resolve_metric(
        metric_name: str,
        train_loss: float | None,
        valid_metrics: Mapping[str, float] | None,
    ):
        if metric_name in {"train", "train_loss"}:
            return "train_loss", train_loss
        if metric_name in {"valid", "val", "valid_loss", "val_loss"}:
            return "valid_loss", aggregate_valid_loss(valid_metrics or {})
        for prefix in ("valid_", "val_"):
            if metric_name.startswith(prefix):
                key = metric_name.removeprefix(prefix)
                return f"valid/{key}", (valid_metrics or {}).get(key)
        if metric_name.startswith("valid:") or metric_name.startswith("valid/"):
            head = metric_name.split(":", 1)[-1].split("/", 1)[-1]
            return f"valid/{head}", (valid_metrics or {}).get(head)
        if valid_metrics and metric_name in valid_metrics:
            return f"valid/{metric_name}", valid_metrics[metric_name]
        return metric_name, None

    def is_improved(current: float, best: float | None) -> bool:
        if best is None:
            return True
        if best_metric_mode == "max":
            return current > best + early_stopping_min_delta
        return current < best - early_stopping_min_delta

    best_value: float | None = None
    epochs_since_improvement = 0
    global_step = initial_global_step
    metrics_history_path = checkpoint_dir / "metrics.jsonl" if checkpoint_dir else None
    if start_epoch > 1:
        if metrics_history_path is None or not metrics_history_path.exists():
            raise FileNotFoundError("Continuation requires existing checkpoint metric history.")
        prior_records = [
            json.loads(line)
            for line in metrics_history_path.read_text().splitlines()
            if line.strip()
        ]
        if not prior_records or int(prior_records[-1]["epoch"]) != start_epoch - 1:
            raise ValueError(f"Metric history does not end at epoch {start_epoch - 1}.")
        if int(prior_records[-1]["global_step"]) != initial_global_step:
            raise ValueError("Metric history global step does not match continuation state.")
        for record in prior_records:
            valid_metrics = _flatten_valid_metrics(record.get("metrics", {}).get("valid", {}))
            _, metric_value = resolve_metric(
                best_metric,
                record.get("train_epoch_loss"),
                valid_metrics,
            )
            if metric_value is not None and math.isfinite(metric_value):
                if is_improved(metric_value, best_value):
                    best_value = metric_value
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
        optimizer_status = (
            f"optimizer state restored from {initial_optimizer_state_path}"
            if initial_optimizer_state_path is not None
            else "optimizer state starts fresh"
        )
        print(
            f"Continuing at epoch {start_epoch} and global step {initial_global_step}; "
            f"{optimizer_status}."
        )

    requested_eval_splits = tuple(dict.fromkeys(str(split) for split in eval_splits))

    print(
        "Train plan: "
        f"{num_train_examples} examples | "
        f"{steps_per_epoch} step(s)/epoch | "
        f"{num_epochs} epoch(s) | "
        f"total step(s) {total_train_steps}"
    )

    with model._device_context:
        replicated_params = _replicate_tree(model._params, devices)
        replicated_state = _replicate_tree(model._state, devices)
        opt_state = _replicate_tree(opt_state, devices)
        strand_reindexing_replicated = _replicate_tree(strand_reindexing, devices)
        stop_training = False

        def evaluate_split(split: str) -> dict[str, dict[str, float]]:
            if split not in data_module._intervals or len(data_module._intervals[split]) == 0:
                return {}
            losses = {head: [] for head in head_names}
            stats_by_head: dict[str, dict[str, Any] | None] = {head: None for head in head_names}
            timing_stats = {
                "batch_wait": 0.0,
                "prepare": 0.0,
                "shard": 0.0,
                "step_dispatch": 0.0,
                "sync": 0.0,
            }

            def prepare_eval_batch(batch_np):
                prep_start = time.perf_counter()
                batch = prepare_batch(batch_np, organism_index_value, head_names)
                prepare_elapsed = time.perf_counter() - prep_start
                shard_start = time.perf_counter()
                batch = _shard_batch(batch, num_devices)
                shard_elapsed = time.perf_counter() - shard_start
                batch["strand_reindexing"] = strand_reindexing_replicated
                return batch, {"prepare": prepare_elapsed, "shard": shard_elapsed}

            batch_iter = iter(
                _prefetch_transformed_iterable(
                    data_module.iter_batches(split, shuffle=False),
                    prefetch_batches,
                    prepare_eval_batch,
                )
            )
            while True:
                wait_start = time.perf_counter()
                try:
                    batch, batch_timing = next(batch_iter)
                except StopIteration:
                    break
                timing_stats["batch_wait"] += time.perf_counter() - wait_start
                timing_stats["prepare"] += batch_timing["prepare"]
                timing_stats["shard"] += batch_timing["shard"]
                step_start = time.perf_counter()
                head_losses, head_stats = eval_step(replicated_params, replicated_state, batch)
                timing_stats["step_dispatch"] += time.perf_counter() - step_start
                for head_name in head_names:
                    sync_start = time.perf_counter()
                    loss_value = float(np.asarray(head_losses[head_name])[0])
                    timing_stats["sync"] += time.perf_counter() - sync_start
                    if not math.isfinite(loss_value):
                        raise FloatingPointError(
                            "Non-finite evaluation loss encountered "
                            f"at split={split}, head={head_name}: loss={loss_value}."
                        )
                    losses[head_name].append(loss_value)
                    stats_by_head[head_name] = _add_stats(
                        stats_by_head[head_name],
                        jax.tree_util.tree_map(
                            lambda value: np.asarray(value)[0],
                            head_stats[head_name],
                        ),
                    )

            split_result: dict[str, dict[str, float]] = {}
            for head_name in head_names:
                head_result = {
                    "loss": float(np.mean(losses[head_name])) if losses[head_name] else float("nan")
                }
                if stats_by_head[head_name] is not None:
                    head_result.update(_finalize_r2_stats(stats_by_head[head_name]))
                split_result[head_name] = head_result
            if profile_host_timing:
                print(
                    _format_host_timing_summary(
                        f"{split} host timing",
                        timing_stats,
                        len(losses[head_names[0]]) if head_names else 0,
                    )
                )
            return split_result

        for epoch in range(start_epoch, num_epochs + 1):
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"{'=' * 60}")
            else:
                print(f"Epoch {epoch}/{num_epochs}")

            epoch_step = 0
            train_loss_sum = None
            train_head_loss_sums = {head_name: None for head_name in head_names}
            train_stats_by_head = {head_name: None for head_name in head_names}
            timing_stats = {
                "batch_wait": 0.0,
                "prepare": 0.0,
                "shard": 0.0,
                "step_dispatch": 0.0,
                "sync": 0.0,
            }

            def prepare_train_batch(batch_np):
                prep_start = time.perf_counter()
                batch = prepare_batch(batch_np, organism_index_value, head_names)
                prepare_elapsed = time.perf_counter() - prep_start
                shard_start = time.perf_counter()
                batch = _shard_batch(batch, num_devices)
                shard_elapsed = time.perf_counter() - shard_start
                batch["strand_reindexing"] = strand_reindexing_replicated
                return batch, {"prepare": prepare_elapsed, "shard": shard_elapsed}

            batch_iter = iter(
                _prefetch_transformed_iterable(
                    data_module.iter_batches("train", seed=seed + epoch),
                    prefetch_batches,
                    prepare_train_batch,
                )
            )
            while True:
                wait_start = time.perf_counter()
                try:
                    batch, batch_timing = next(batch_iter)
                except StopIteration:
                    break
                timing_stats["batch_wait"] += time.perf_counter() - wait_start
                timing_stats["prepare"] += batch_timing["prepare"]
                timing_stats["shard"] += batch_timing["shard"]
                step_start = time.perf_counter()
                (
                    replicated_params,
                    opt_state,
                    loss_value,
                    head_losses,
                    head_stats,
                ) = train_step(
                    replicated_params,
                    replicated_state,
                    opt_state,
                    batch,
                )
                timing_stats["step_dispatch"] += time.perf_counter() - step_start
                loss_replica = loss_value[0]
                train_loss_sum = (
                    loss_replica if train_loss_sum is None else train_loss_sum + loss_replica
                )
                for head_name in head_names:
                    head_loss_replica = head_losses[head_name][0]
                    train_head_loss_sums[head_name] = (
                        head_loss_replica
                        if train_head_loss_sums[head_name] is None
                        else train_head_loss_sums[head_name] + head_loss_replica
                    )
                    train_stats_by_head[head_name] = _add_device_stats(
                        train_stats_by_head[head_name],
                        jax.tree_util.tree_map(
                            lambda value: value[0],
                            head_stats[head_name],
                        ),
                    )
                epoch_step += 1
                global_step += 1

                should_sync_step = (
                    epoch_step == 1
                    or epoch_step % progress_interval == 0
                    or epoch_step == steps_per_epoch
                    or global_step >= total_train_steps
                )
                if should_sync_step:
                    sync_start = time.perf_counter()
                    loss_scalar = float(np.asarray(loss_replica))
                    if profile_host_timing:
                        timing_stats["sync"] += time.perf_counter() - sync_start
                    if not math.isfinite(loss_scalar):
                        raise FloatingPointError(
                            "Non-finite training loss encountered "
                            f"at epoch={epoch}, epoch_step={epoch_step}, "
                            f"global_step={global_step}: loss={loss_scalar}."
                        )
                    if verbose:
                        print(
                            f"  step {epoch_step:0{step_width}d}/{steps_per_epoch:0{step_width}d}"
                            f" | loss {loss_scalar:.4f}",
                            end="\r",
                            flush=True,
                        )

                    if use_wandb:
                        wandb.log(
                            {
                                "step/train_loss": loss_scalar,
                                "train/step_loss": loss_scalar,
                                "epoch": epoch,
                                "step": global_step,
                            }
                        )

                if global_step >= total_train_steps:
                    stop_training = True
                    break

            train_loss_avg = None
            if train_loss_sum is not None and epoch_step > 0:
                train_loss_avg = float(np.asarray(train_loss_sum / epoch_step))
                if not math.isfinite(train_loss_avg):
                    raise FloatingPointError(
                        "Non-finite training loss encountered "
                        f"at epoch={epoch}: loss={train_loss_avg}."
                    )
            if verbose:
                print()
            if profile_host_timing and epoch_step > 0:
                print(
                    _format_host_timing_summary(
                        "train host timing",
                        timing_stats,
                        epoch_step,
                    )
                )
            if train_loss_avg is not None:
                print(f"  Train loss: {train_loss_avg:.4f}")
                if use_wandb:
                    wandb.log(
                        {
                            "epoch/train_loss": train_loss_avg,
                            "train/epoch_loss": train_loss_avg,
                            "epoch": epoch,
                        }
                    )

            split_metrics: dict[str, dict[str, dict[str, float]]] = {}
            if "train" in requested_eval_splits and epoch_step > 0:
                train_metrics: dict[str, dict[str, float]] = {}
                for head_name in head_names:
                    head_loss_sum = train_head_loss_sums[head_name]
                    head_result = {
                        "loss": (
                            float(np.asarray(head_loss_sum / epoch_step))
                            if head_loss_sum is not None
                            else float("nan")
                        )
                    }
                    if train_stats_by_head[head_name] is not None:
                        head_result.update(_finalize_r2_stats(train_stats_by_head[head_name]))
                    train_metrics[head_name] = head_result
                split_metrics["train"] = train_metrics

            for split in requested_eval_splits:
                metrics = split_metrics.get(split) if split == "train" else evaluate_split(split)
                if not metrics:
                    continue
                split_metrics[split] = metrics
                printable = []
                for head_name, head_result in metrics.items():
                    printable.append(
                        f"{head_name}: "
                        f"loss={head_result['loss']:.4f}, "
                        f"r2_global={head_result['r2_global']:.4f}, "
                        f"r2_over_loci={head_result['r2_over_loci']:.4f}, "
                        f"r2_over_cell_types={head_result['r2_over_cell_types']:.4f}, "
                        f"differential_pearson_r={head_result['differential_pearson_r']:.4f}, "
                        f"double_centered_r2={head_result['double_centered_r2']:.4f}"
                    )
                print(f"  {split.capitalize()} metrics:", "; ".join(printable))
                if use_wandb:
                    split_log = {"epoch": epoch}
                    split_loss = float(sum(head_result["loss"] for head_result in metrics.values()))
                    split_log[f"epoch/{split}_loss"] = split_loss
                    for head_name, head_result in metrics.items():
                        for metric_name, metric_value in head_result.items():
                            split_log[f"{split}/{head_name}/{metric_name}"] = metric_value
                            split_log[f"epoch/{split}/{head_name}/{metric_name}"] = metric_value
                    if len(metrics) == 1:
                        head_result = next(iter(metrics.values()))
                        for metric_name, metric_value in head_result.items():
                            split_log[f"epoch/{split}_{metric_name}"] = metric_value
                    split_log[f"{split}/loss"] = split_loss
                    wandb.log(split_log)

            valid_metrics: Mapping[str, float] | None = None
            if "valid" in split_metrics:
                valid_metrics = _flatten_valid_metrics(split_metrics["valid"])

            metric_label, metric_value = resolve_metric(best_metric, train_loss_avg, valid_metrics)
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_epoch_loss": train_loss_avg,
                "metrics": split_metrics,
            }
            if metrics_history_path:
                _append_jsonl(metrics_history_path, epoch_record)
            if metric_value is not None and math.isfinite(metric_value):
                if is_improved(metric_value, best_value):
                    best_value = metric_value
                    epochs_since_improvement = 0
                    if use_wandb:
                        wandb.log({"best/" + metric_label: metric_value, "epoch": epoch})
                    if checkpoint_dir:
                        model._params = _unreplicate_tree(replicated_params)
                        model._state = _unreplicate_tree(replicated_state)
                        print(
                            f"  Metric improved ({metric_label} = {metric_value:.4f}) "
                            " -- saving best checkpoint"
                        )
                        model.save_checkpoint(
                            checkpoint_dir / "best",
                            save_full_model=False,
                            save_lora_adapters=train_lora,
                        )
                        _save_optimizer_state(
                            checkpoint_dir / "best" / "optimizer_state",
                            _unreplicate_tree(opt_state),
                        )
                        _write_json(checkpoint_dir / "best" / "metrics.json", epoch_record)
                else:
                    epochs_since_improvement += 1
            else:
                print(f"  Best metric ({metric_label}): unavailable")

            if checkpoint_dir:
                model._params = _unreplicate_tree(replicated_params)
                model._state = _unreplicate_tree(replicated_state)
                model.save_checkpoint(
                    checkpoint_dir / "last",
                    save_full_model=False,
                    save_lora_adapters=train_lora,
                )
                _save_optimizer_state(
                    checkpoint_dir / "last" / "optimizer_state",
                    _unreplicate_tree(opt_state),
                )
                _write_json(checkpoint_dir / "last" / "metrics.json", epoch_record)

            if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
                print(f"\n  Early stopping: no improvement for {epochs_since_improvement} epoch(s)")
                break
            if stop_training:
                print(f"  Reached requested training steps: {global_step}/{total_train_steps}")
                break

        model._params = _unreplicate_tree(replicated_params)
        model._state = _unreplicate_tree(replicated_state)

    if checkpoint_dir and not (checkpoint_dir / "last").exists():
        model.save_checkpoint(
            checkpoint_dir / "last",
            save_full_model=False,
            save_lora_adapters=train_lora,
        )
        _save_optimizer_state(
            checkpoint_dir / "last" / "optimizer_state",
            _unreplicate_tree(opt_state),
        )

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")

    if use_wandb:
        wandb.finish()


__all__ = [
    "register_predefined_heads",
    "create_optimizer",
    "train",
]
