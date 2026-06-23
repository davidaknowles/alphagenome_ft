"""FP8-ready LoRA adapters for AlphaGenome backbone projections."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Literal

import haiku as hk
import jax.numpy as jnp
from jaxtyping import Array, Float

ParamDType = Literal["float32", "fp32", "bfloat16", "bf16", "float16", "fp16", "fp8"]
ComputeDType = Literal["input", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16", "fp8"]


DEFAULT_BACKBONE_LORA_TARGETS: tuple[str, ...] = (
    "q_layer",
    "k_layer",
    "v_layer",
    "linear_embedding",
    "linear_k",
    "linear_q",
    "linear_v",
    "linear_pos_features",
    "linear_y_q",
    "linear_y_k",
    "linear_pair",
)


@dataclass(frozen=True)
class BackboneLoRAConfig:
    """Configuration for transformer-backbone LoRA adapters.

    Attributes:
        rank: Rank of the low-rank adapter.
        alpha: LoRA scaling factor. The effective multiplier is ``alpha / rank``.
        fp8_enabled: If True, require Transformer Engine and run adapter matmuls under
            its JAX FP8 autocast context. This is a compatibility shortcut for
            ``lora_compute_dtype="fp8"``.
        base_param_dtype: Storage dtype for newly initialized base projection weights.
            ``"fp8"`` stores base ``w`` leaves as JAX float8 and dequantizes them
            to the requested compute dtype before matmul. Loaded checkpoint leaves
            keep their checkpoint dtype until the runtime cast step.
        lora_param_dtype: Storage dtype for trainable LoRA adapter leaves.
        activation_dtype: Activation dtype to feed into patched linear GEMMs. ``"fp8"``
            means "let Transformer Engine quantize inputs inside FP8 GEMMs"; persistent
            activations are not stored as FP8 JAX arrays.
        base_compute_dtype: Compute precision for the frozen base projection.
        lora_compute_dtype: Compute precision for LoRA adapter projections. If unset,
            defaults to ``"fp8"`` when ``fp8_enabled=True`` and otherwise to
            ``activation_dtype``.
        target_names: Haiku ``hk.Linear`` module names to adapt while building the
            AlphaGenome trunk. Names are matched exactly.
    """

    rank: int = 16
    alpha: float = 16.0
    fp8_enabled: bool = False
    base_param_dtype: ParamDType = "float32"
    lora_param_dtype: ParamDType = "float32"
    activation_dtype: ComputeDType = "bfloat16"
    base_compute_dtype: ComputeDType = "bfloat16"
    lora_compute_dtype: ComputeDType | None = None
    target_names: Sequence[str] = DEFAULT_BACKBONE_LORA_TARGETS

    def normalized_target_names(self) -> frozenset[str]:
        return frozenset(str(name) for name in self.target_names)

    def resolved_lora_compute_dtype(self) -> ComputeDType:
        if self.lora_compute_dtype is not None:
            return self.lora_compute_dtype
        return "fp8" if self.fp8_enabled else self.activation_dtype

    def uses_fp8(self) -> bool:
        return (
            self.base_compute_dtype == "fp8"
            or self.resolved_lora_compute_dtype() == "fp8"
        )

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}.")
        _resolve_param_dtype(
            self.base_param_dtype,
            field_name="base_param_dtype",
            allow_fp8=True,
        )
        _resolve_param_dtype(self.lora_param_dtype, field_name="lora_param_dtype")
        _resolve_compute_dtype(self.activation_dtype, field_name="activation_dtype")
        _resolve_compute_dtype(self.base_compute_dtype, field_name="base_compute_dtype")
        _resolve_compute_dtype(
            self.resolved_lora_compute_dtype(),
            field_name="lora_compute_dtype",
        )
        if self.activation_dtype == "fp8" and not self.uses_fp8():
            raise ValueError("activation_dtype='fp8' requires an FP8 base or LoRA compute path.")
        if self.uses_fp8() and self.rank % 16 != 0:
            raise ValueError(
                "FP8 LoRA rank must be a multiple of 16 because Transformer "
                "Engine quantized GEMMs require contracting dimensions aligned "
                f"to 16. Got rank={self.rank}."
            )


def _default_w_init(input_size: int):
    return hk.initializers.TruncatedNormal(stddev=1.0 / math.sqrt(input_size))


def _canonical_dtype_name(value: str, *, field_name: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "float32": "float32",
        "fp32": "float32",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
        "input": "input",
        "fp8": "fp8",
    }
    if normalized not in aliases:
        raise ValueError(
            f"{field_name} must be one of input, float32/fp32, bfloat16/bf16, "
            f"float16/fp16, or fp8; got {value!r}."
        )
    return aliases[normalized]


def _resolve_fp8_param_dtype():
    try:
        return jnp.float8_e4m3fn
    except AttributeError as exc:  # pragma: no cover - depends on installed JAX.
        raise ValueError(
            "fp8 parameter storage requires a JAX build exposing "
            "`jax.numpy.float8_e4m3fn`."
        ) from exc


def _is_fp8_storage_dtype(dtype) -> bool:
    try:
        return dtype == _resolve_fp8_param_dtype()
    except ValueError:
        return False


def _resolve_param_dtype(value: str, *, field_name: str, allow_fp8: bool = False):
    normalized = _canonical_dtype_name(value, field_name=field_name)
    if normalized == "input":
        raise ValueError(
            f"{field_name} cannot be {value!r}; parameter storage must be fp32/bf16/fp16"
            " or fp8 where explicitly supported."
        )
    if normalized == "fp8":
        if not allow_fp8:
            raise ValueError(
                f"{field_name} cannot be {value!r}; only base projection weights "
                "support fp8 storage."
            )
        return _resolve_fp8_param_dtype()
    return {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }[normalized]


def _resolve_compute_dtype(value: str, *, field_name: str):
    normalized = _canonical_dtype_name(value, field_name=field_name)
    if normalized in {"input", "fp8"}:
        return normalized
    return {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }[normalized]


def _cast_for_compute(value, precision: ComputeDType, *, fallback_dtype):
    dtype = _resolve_compute_dtype(precision, field_name="compute_dtype")
    if hasattr(value, "dtype") and _is_fp8_storage_dtype(value.dtype):
        if dtype not in {"input", "fp8"}:
            return value.astype(dtype)
        return value.astype(fallback_dtype)
    if dtype == "fp8":
        return value.astype(fallback_dtype)
    if dtype == "input":
        return value
    return value.astype(dtype)


def _validate_fp8_contracting_dim(size: int, *, label: str) -> None:
    if size % 16 != 0:
        raise ValueError(
            f"FP8 {label} contracting dimension must be a multiple of 16 for "
            f"Transformer Engine quantized GEMMs. Got {size}."
        )


def _require_transformer_engine():
    try:
        import transformer_engine.jax as te_jax  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional CUDA package.
        raise ImportError(
            "FP8 LoRA requires `transformer_engine[jax]`. Install it in the active "
            "JAX environment, e.g. `pip install --no-build-isolation "
            "'transformer-engine[jax]'`."
        ) from exc
    return te_jax


def _dense(lhs, rhs, *, precision: ComputeDType, label: str):
    """Run a dense projection with selectable precision."""
    if _canonical_dtype_name(precision, field_name="precision") != "fp8":
        lhs_compute = _cast_for_compute(lhs, precision, fallback_dtype=lhs.dtype)
        rhs_compute = _cast_for_compute(rhs, precision, fallback_dtype=lhs_compute.dtype)
        return jnp.dot(lhs_compute, rhs_compute)

    _validate_fp8_contracting_dim(lhs.shape[-1], label=label)
    lhs = _cast_for_compute(lhs, "input", fallback_dtype=lhs.dtype)
    rhs = _cast_for_compute(rhs, "input", fallback_dtype=lhs.dtype)

    te_jax = _require_transformer_engine()
    from transformer_engine.jax import dense as te_dense  # type: ignore
    from transformer_engine.jax.quantize import QuantizerFactory  # type: ignore

    fp8_autocast = getattr(te_jax, "fp8_autocast", None)
    if fp8_autocast is None:  # pragma: no cover - optional package API guard.
        raise RuntimeError(
            "Installed transformer_engine.jax does not expose fp8_autocast; "
            "cannot enable FP8 LoRA."
        )

    try:
        context = fp8_autocast(enabled=True)
    except TypeError:  # pragma: no cover - API-version compatibility guard.
        context = fp8_autocast()
    with context:
        quantizer_set = QuantizerFactory.create_set()
        return te_dense.dense(
            lhs,
            rhs,
            contracting_dims=((lhs.ndim - 1,), (0,)),
            quantizer_set=quantizer_set,
        )


class LinearWithLoRA(hk.Module):
    """Drop-in ``hk.Linear`` replacement with trainable LoRA adapter leaves."""

    def __init__(
        self,
        output_size: int,
        config: BackboneLoRAConfig,
        *,
        with_bias: bool = True,
        w_init=None,
        b_init=None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        config.validate()
        self._output_size = int(output_size)
        self._config = config
        self._with_bias = with_bias
        self._w_init = w_init
        self._b_init = b_init

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... O"]:
        input_size = x.shape[-1]
        activation = _cast_for_compute(
            x,
            self._config.activation_dtype,
            fallback_dtype=x.dtype,
        )
        base_param_dtype = _resolve_param_dtype(
            self._config.base_param_dtype,
            field_name="base_param_dtype",
            allow_fp8=True,
        )
        lora_param_dtype = _resolve_param_dtype(
            self._config.lora_param_dtype,
            field_name="lora_param_dtype",
        )
        w_init = self._w_init or _default_w_init(input_size)
        w = hk.get_parameter(
            "w",
            shape=(input_size, self._output_size),
            dtype=base_param_dtype,
            init=w_init,
        )
        y = _dense(
            activation,
            w,
            precision=self._config.base_compute_dtype,
            label="base",
        )

        if self._with_bias:
            b_init = self._b_init or jnp.zeros
            bias_dtype = "bfloat16" if _is_fp8_storage_dtype(base_param_dtype) else self._config.base_param_dtype
            b = hk.get_parameter(
                "b",
                shape=(self._output_size,),
                dtype=_resolve_param_dtype(
                    bias_dtype,
                    field_name="base_bias_dtype",
                ),
                init=b_init,
            )
            y = y + b.astype(y.dtype)

        rank = self._config.rank
        lora_a = hk.get_parameter(
            "lora_a",
            shape=(input_size, rank),
            dtype=lora_param_dtype,
            init=hk.initializers.RandomNormal(stddev=0.01),
        )
        lora_b = hk.get_parameter(
            "lora_b",
            shape=(rank, self._output_size),
            dtype=lora_param_dtype,
            init=hk.initializers.Constant(0.0),
        )
        lora_precision = self._config.resolved_lora_compute_dtype()
        delta = _dense(
            activation,
            lora_a,
            precision=lora_precision,
            label="LoRA A",
        )
        delta = _dense(
            delta,
            lora_b,
            precision=lora_precision,
            label="LoRA B",
        )
        return y + delta.astype(y.dtype) * (self._config.alpha / rank)


@contextmanager
def patch_haiku_linear(config: BackboneLoRAConfig) -> Iterator[None]:
    """Patch ``hk.Linear`` so selected AlphaGenome trunk projections get LoRA.

    The patch is deliberately scoped by context manager. It should wrap only the
    AlphaGenome trunk construction/application, not user heads.
    """

    original_linear = hk.Linear
    target_names = config.normalized_target_names()

    def maybe_lora_linear(
        output_size,
        with_bias: bool = True,
        w_init=None,
        b_init=None,
        name: str | None = None,
    ):
        if name is not None and str(name) in target_names:
            return LinearWithLoRA(
                output_size,
                config,
                with_bias=with_bias,
                w_init=w_init,
                b_init=b_init,
                name=name,
            )
        return original_linear(
            output_size,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            name=name,
        )

    hk.Linear = maybe_lora_linear  # type: ignore[assignment]
    try:
        yield
    finally:
        hk.Linear = original_linear  # type: ignore[assignment]


def parse_lora_target_names(raw: str | None) -> tuple[str, ...]:
    """Parse comma-separated target names from CLI/env settings."""
    if raw is None or raw.strip().lower() in {"", "default", "all"}:
        return DEFAULT_BACKBONE_LORA_TARGETS
    return tuple(item.strip() for item in raw.split(",") if item.strip())


__all__ = [
    "BackboneLoRAConfig",
    "DEFAULT_BACKBONE_LORA_TARGETS",
    "LinearWithLoRA",
    "parse_lora_target_names",
    "patch_haiku_linear",
]
