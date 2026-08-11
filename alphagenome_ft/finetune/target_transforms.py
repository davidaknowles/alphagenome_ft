"""Transforms for optimization-scale genomic targets and raw-scale evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol, Sequence

import jax.numpy as jnp
import numpy as np


def _validate_knots(
    source: Sequence[Sequence[float]],
    transformed: Sequence[Sequence[float]],
) -> None:
    if len(source) != len(transformed) or not source:
        raise ValueError("Source and transformed knots must have the same nonzero channel count.")
    for channel, (source_channel, transformed_channel) in enumerate(zip(source, transformed)):
        if len(source_channel) != len(transformed_channel) or len(source_channel) < 2:
            raise ValueError(f"Transform channel {channel} must have at least two paired knots.")
        if np.any(np.diff(source_channel) <= 0):
            raise ValueError(f"Source knots for channel {channel} must be strictly increasing.")
        if np.any(np.diff(transformed_channel) <= 0):
            raise ValueError(
                f"Transformed knots for channel {channel} must be strictly increasing."
            )


@dataclass(frozen=True)
class PiecewiseLinearTargetTransform:
    """Per-channel monotone transform with an explicit inverse."""

    source_knots: tuple[np.ndarray, ...]
    transformed_knots: tuple[np.ndarray, ...]

    @classmethod
    def from_path(cls, path: Path) -> "PiecewiseLinearTargetTransform":
        payload = json.loads(Path(path).read_text())
        if payload.get("kind") != "piecewise_linear":
            raise ValueError(
                f"Unsupported target transform kind in {path}: {payload.get('kind')!r}"
            )
        source = payload["source_knots"]
        transformed = payload["transformed_knots"]
        _validate_knots(source, transformed)
        return cls(
            source_knots=tuple(np.asarray(values, dtype=np.float32) for values in source),
            transformed_knots=tuple(np.asarray(values, dtype=np.float32) for values in transformed),
        )

    @property
    def num_channels(self) -> int:
        return len(self.source_knots)

    def forward_numpy(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != self.num_channels:
            raise ValueError(
                f"Transform has {self.num_channels} channels, received shape {values.shape}."
            )
        result = np.empty(values.shape, dtype=np.float32)
        for channel, (source, transformed) in enumerate(
            zip(self.source_knots, self.transformed_knots)
        ):
            result[..., channel] = np.interp(values[..., channel], source, transformed).astype(
                np.float32
            )
        return result

    def forward_jax(self, values):
        if values.shape[-1] != self.num_channels:
            raise ValueError(
                f"Transform has {self.num_channels} channels, received shape {values.shape}."
            )
        channels = []
        for channel, (source, transformed) in enumerate(
            zip(self.source_knots, self.transformed_knots)
        ):
            channels.append(
                jnp.interp(
                    values[..., channel],
                    jnp.asarray(source),
                    jnp.asarray(transformed),
                )
            )
        return jnp.stack(channels, axis=-1)

    def inverse_jax(self, values):
        if values.shape[-1] != self.num_channels:
            raise ValueError(
                f"Transform has {self.num_channels} channels, received shape {values.shape}."
            )
        channels = []
        for channel, (source, transformed) in enumerate(
            zip(self.source_knots, self.transformed_knots)
        ):
            channels.append(
                jnp.interp(
                    values[..., channel],
                    jnp.asarray(transformed),
                    jnp.asarray(source),
                )
            )
        return jnp.stack(channels, axis=-1)


@dataclass(frozen=True)
class SpatialRebinTargetTransform:
    """Locally rebin base-expanded tracks while retaining their sequence length."""

    width: int
    output_scale: float = 1.0

    @classmethod
    def from_path(cls, path: Path) -> "SpatialRebinTargetTransform":
        payload = json.loads(Path(path).read_text())
        if payload.get("kind") != "spatial_rebin":
            raise ValueError(
                f"Unsupported target transform kind in {path}: {payload.get('kind')!r}"
            )
        width = int(payload["width"])
        output_scale = float(payload.get("output_scale", 1.0))
        if width <= 0 or output_scale <= 0:
            raise ValueError("Spatial rebin width and output scale must be positive.")
        return cls(width=width, output_scale=output_scale)

    @staticmethod
    def _full_length(length: int, width: int) -> int:
        return length - length % width

    def forward_numpy(self, values: np.ndarray) -> np.ndarray:
        full_length = self._full_length(values.shape[-2], self.width)
        chunks = []
        if full_length:
            prefix = values[..., :full_length, :].reshape(
                *values.shape[:-2], full_length // self.width, self.width, values.shape[-1]
            )
            chunks.append(np.repeat(prefix.mean(axis=-2), self.width, axis=-2))
        if full_length < values.shape[-2]:
            tail = values[..., full_length:, :].mean(axis=-2, keepdims=True)
            chunks.append(np.repeat(tail, values.shape[-2] - full_length, axis=-2))
        return (np.concatenate(chunks, axis=-2) * self.output_scale).astype(np.float32)

    def forward_jax(self, values):
        full_length = self._full_length(values.shape[-2], self.width)
        chunks = []
        if full_length:
            prefix = values[..., :full_length, :].reshape(
                *values.shape[:-2], full_length // self.width, self.width, values.shape[-1]
            )
            chunks.append(jnp.repeat(jnp.mean(prefix, axis=-2), self.width, axis=-2))
        if full_length < values.shape[-2]:
            tail = jnp.mean(values[..., full_length:, :], axis=-2, keepdims=True)
            chunks.append(jnp.repeat(tail, values.shape[-2] - full_length, axis=-2))
        return jnp.concatenate(chunks, axis=-2) * self.output_scale

    def inverse_jax(self, values):
        # Spatial averaging is not invertible. Undo the units change for raw-scale metrics.
        return values / self.output_scale


class TargetTransform(Protocol):
    def forward_jax(self, values): ...

    def inverse_jax(self, values): ...


def load_target_transform(path: Path) -> TargetTransform:
    kind = json.loads(Path(path).read_text()).get("kind")
    if kind == "piecewise_linear":
        return PiecewiseLinearTargetTransform.from_path(path)
    if kind == "spatial_rebin":
        return SpatialRebinTargetTransform.from_path(path)
    raise ValueError(f"Unsupported target transform kind in {path}: {kind!r}")


__all__ = [
    "PiecewiseLinearTargetTransform",
    "SpatialRebinTargetTransform",
    "TargetTransform",
    "load_target_transform",
]
