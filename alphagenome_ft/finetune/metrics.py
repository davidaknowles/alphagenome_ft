"""Shared finetuning metrics.

These helpers are backend-neutral and operate on NumPy-compatible arrays.  The
JAX training loop has JIT-friendly equivalents, but these functions define the
shared metric semantics used by backend adapters and offline evaluation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _maybe_bin_128bp(values: np.ndarray) -> np.ndarray:
    if values.ndim != 3:
        return values
    loci = values.shape[1]
    if loci >= 8192 and loci % 128 == 0:
        return values.reshape(values.shape[0], loci // 128, 128, values.shape[2]).mean(axis=2)
    return values


def _double_center_pearson(prediction: np.ndarray, targets: np.ndarray) -> float:
    prediction = _maybe_bin_128bp(prediction)
    targets = _maybe_bin_128bp(targets)
    pred_matrix = prediction.reshape(-1, prediction.shape[-1]).astype(np.float32)
    true_matrix = targets.reshape(-1, targets.shape[-1]).astype(np.float32)
    pred_centered = (
        pred_matrix
        - pred_matrix.mean(axis=0, keepdims=True)
        - pred_matrix.mean(axis=1, keepdims=True)
        + pred_matrix.mean()
    )
    true_centered = (
        true_matrix
        - true_matrix.mean(axis=0, keepdims=True)
        - true_matrix.mean(axis=1, keepdims=True)
        + true_matrix.mean()
    )
    pred_flat = pred_centered.reshape(-1)
    true_flat = true_centered.reshape(-1)
    pred_std = float(np.sqrt(np.sum(np.square(pred_flat))))
    true_std = float(np.sqrt(np.sum(np.square(true_flat))))
    if pred_std <= 0 or true_std <= 0:
        return float("nan")
    return float(np.sum(pred_flat * true_flat) / (pred_std * true_std))


def select_prediction_for_targets(prediction: Any, targets: np.ndarray) -> np.ndarray:
    """Return the prediction array matching the target tensor shape."""
    if hasattr(prediction, "shape"):
        return np.asarray(prediction)
    if not isinstance(prediction, Mapping):
        raise TypeError(f"Unsupported prediction type for R2 metrics: {type(prediction)!r}")

    preferred_keys = (
        "predictions_1bp",
        "predictions",
        "scaled_predictions_1bp",
    )
    for key in preferred_keys:
        value = prediction.get(key)
        if hasattr(value, "shape") and tuple(value.shape) == tuple(targets.shape):
            return np.asarray(value)

    for key, value in prediction.items():
        if (
            str(key).startswith(("predictions_", "scaled_predictions_"))
            and hasattr(value, "shape")
            and tuple(value.shape) == tuple(targets.shape)
        ):
            return np.asarray(value)

    shapes = {str(key): getattr(value, "shape", None) for key, value in prediction.items()}
    raise ValueError(
        "Could not find a prediction array matching target shape "
        f"{targets.shape}; available prediction shapes: {shapes}"
    )


def r2_metrics(prediction: Any, targets: Any) -> dict[str, float]:
    """Compute R2 globally, over loci, and over cell types/tracks.

    The expected shape is ``[batch, loci, tracks]``.  This matches the JAX
    training loop's ``r2_global``, ``r2_over_loci``, and ``r2_over_cell_types``
    semantics.
    """
    y = np.asarray(targets, dtype=np.float32)
    yhat = select_prediction_for_targets(prediction, y).astype(np.float32, copy=False)
    residual = yhat - y

    sse = float(np.sum(np.square(residual)))
    centered = y - float(np.mean(y))
    sst = float(np.sum(np.square(centered)))
    r2_global = float(1.0 - sse / sst) if sst > 0 else float("nan")

    target_mean_by_locus = np.mean(y, axis=-1, keepdims=True)
    differential_pearson_r = _double_center_pearson(yhat, y)

    sst_by_locus = np.sum(np.square(y - target_mean_by_locus), axis=-1)
    sse_by_locus = np.sum(np.square(residual), axis=-1)
    valid_loci = sst_by_locus > 0
    r2_over_loci = (
        float(np.mean(1.0 - sse_by_locus[valid_loci] / sst_by_locus[valid_loci]))
        if np.any(valid_loci)
        else float("nan")
    )

    target_mean_by_track = np.mean(y, axis=(0, 1), keepdims=False)
    sst_by_track = np.sum(np.square(y - target_mean_by_track.reshape((1, 1, -1))), axis=(0, 1))
    sse_by_track = np.sum(np.square(residual), axis=(0, 1))
    valid_tracks = sst_by_track > 0
    r2_over_cell_types = (
        float(np.mean(1.0 - sse_by_track[valid_tracks] / sst_by_track[valid_tracks]))
        if np.any(valid_tracks)
        else float("nan")
    )

    return {
        "r2_global": r2_global,
        "r2_over_loci": r2_over_loci,
        "r2_over_cell_types": r2_over_cell_types,
        "differential_pearson_r": differential_pearson_r,
    }


__all__ = ["select_prediction_for_targets", "r2_metrics"]
