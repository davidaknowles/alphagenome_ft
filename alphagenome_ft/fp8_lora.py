"""FP8/NVFP4-ready LoRA and LoCon adapters for AlphaGenome backbones."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import math
from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

ParamDType = Literal[
    "float32",
    "fp32",
    "bfloat16",
    "bf16",
    "float16",
    "fp16",
    "fp8",
    "fp4",
    "nvfp4",
]
ComputeDType = Literal[
    "input",
    "float32",
    "fp32",
    "bfloat16",
    "bf16",
    "float16",
    "fp16",
    "fp8",
    "fp4",
    "nvfp4",
]


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

DEFAULT_BACKBONE_LOCON_TARGETS: tuple[str, ...] = (
    "downres_block_4",
    "downres_block_5",
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
        fp4_enabled: If True, require Transformer Engine and run adapter matmuls with
            the NVFP4 block-scaling recipe. This is a compatibility shortcut for
            ``lora_compute_dtype="fp4"``.
        base_param_dtype: Storage dtype for newly initialized base projection weights.
            ``"fp8"`` stores base ``w`` leaves as JAX float8 and dequantizes them
            to the requested compute dtype before matmul. Loaded checkpoint leaves
            keep their checkpoint dtype until the runtime cast step. ``"fp4"`` stores
            base ``w`` leaves as JAX E2M1 FP4 and dequantizes to the requested compute
            dtype before matmul. This is experimental storage, not full block-scaled
            NVFP4 tensor storage with separate scale metadata.
        lora_param_dtype: Storage dtype for trainable LoRA adapter leaves.
        activation_dtype: Activation dtype to feed into patched linear GEMMs.
            ``"fp8"``/``"fp4"`` means "let Transformer Engine quantize inputs inside
            quantized GEMMs"; persistent activations are not stored as FP8/FP4 arrays.
        base_compute_dtype: Compute precision for the frozen base projection.
        lora_compute_dtype: Compute precision for LoRA adapter projections. If unset,
            defaults to ``"fp4"`` when ``fp4_enabled=True``, ``"fp8"`` when
            ``fp8_enabled=True``, and otherwise to ``activation_dtype``.
        target_names: Haiku ``hk.Linear`` module names to adapt while building the
            AlphaGenome trunk. Names are matched exactly.
    """

    rank: int = 16
    alpha: float = 16.0
    fp8_enabled: bool = False
    fp4_enabled: bool = False
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
        if self.fp4_enabled:
            return "fp4"
        return "fp8" if self.fp8_enabled else self.activation_dtype

    def uses_fp8(self) -> bool:
        return (
            _canonical_dtype_name(self.base_compute_dtype, field_name="base_compute_dtype")
            == "fp8"
            or _canonical_dtype_name(
                self.resolved_lora_compute_dtype(),
                field_name="lora_compute_dtype",
            )
            == "fp8"
        )

    def uses_fp4(self) -> bool:
        return (
            _canonical_dtype_name(self.base_compute_dtype, field_name="base_compute_dtype")
            == "fp4"
            or _canonical_dtype_name(
                self.resolved_lora_compute_dtype(),
                field_name="lora_compute_dtype",
            )
            == "fp4"
        )

    def uses_transformer_engine(self) -> bool:
        return self.uses_fp8() or self.uses_fp4()

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}.")
        if self.fp8_enabled and self.fp4_enabled:
            raise ValueError("fp8_enabled and fp4_enabled are mutually exclusive.")
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
        activation_dtype = _canonical_dtype_name(
            self.activation_dtype,
            field_name="activation_dtype",
        )
        if activation_dtype in {"fp8", "fp4"} and not self.uses_transformer_engine():
            raise ValueError(
                f"activation_dtype={self.activation_dtype!r} requires an FP8/FP4 base "
                "or LoRA compute path."
            )
        rank_alignment = 32 if self.uses_fp4() else 16
        if self.uses_transformer_engine() and self.rank % rank_alignment != 0:
            raise ValueError(
                "Transformer Engine LoRA rank must be a multiple of "
                f"{rank_alignment} because quantized GEMMs require contracting "
                f"dimensions aligned to {rank_alignment}. Got rank={self.rank}."
            )


@dataclass(frozen=True)
class BackboneLoConConfig:
    """Configuration for convolutional LoRA adapters.

    LoCon applies a trainable low-rank convolutional residual to selected
    ``StandardizedConv1D`` modules in the AlphaGenome convolutional trunk. The
    original convolution parameters keep their normal names, while adapter
    weights are stored as ``locon_down_w`` and ``locon_up_w`` leaves.
    """

    rank: int = 4
    alpha: float = 1.0
    param_dtype: ParamDType = "float32"
    compute_dtype: ComputeDType = "bfloat16"
    target_names: Sequence[str] = DEFAULT_BACKBONE_LOCON_TARGETS

    def normalized_target_names(self) -> frozenset[str]:
        return frozenset(str(name) for name in self.target_names)

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError(f"LoCon rank must be positive, got {self.rank}.")
        _resolve_param_dtype(self.param_dtype, field_name="locon_param_dtype")
        _resolve_compute_dtype(self.compute_dtype, field_name="locon_compute_dtype")


def _default_w_init(input_size: int):
    return hk.initializers.TruncatedNormal(stddev=1.0 / math.sqrt(input_size))


def _adapter_parameter_rng(parameter_name: str):
    """Return a stable local RNG keyed by module path and adapter leaf name."""
    identity = f"{hk.experimental.current_name()}/{parameter_name}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")
    return jax.random.PRNGKey(seed)


def _low_precision_safe_init(init):
    def wrapped(shape, dtype):
        if _is_low_precision_storage_dtype(dtype):
            return init(shape, jnp.float32).astype(dtype)
        return init(shape, dtype)

    return wrapped


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
        "fp4": "fp4",
        "nvfp4": "fp4",
    }
    if normalized not in aliases:
        raise ValueError(
            f"{field_name} must be one of input, float32/fp32, bfloat16/bf16, "
            f"float16/fp16, fp8, or fp4/nvfp4; got {value!r}."
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


def _resolve_fp4_param_dtype():
    try:
        return jnp.float4_e2m1fn
    except AttributeError as exc:  # pragma: no cover - depends on installed JAX.
        raise ValueError(
            "fp4 parameter storage requires a JAX build exposing "
            "`jax.numpy.float4_e2m1fn`."
        ) from exc


def _is_fp8_storage_dtype(dtype) -> bool:
    try:
        return dtype == _resolve_fp8_param_dtype()
    except ValueError:
        return False


def _is_fp4_storage_dtype(dtype) -> bool:
    try:
        return dtype == _resolve_fp4_param_dtype()
    except ValueError:
        return False


def _is_low_precision_storage_dtype(dtype) -> bool:
    return _is_fp8_storage_dtype(dtype) or _is_fp4_storage_dtype(dtype)


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
    if normalized == "fp4":
        if not allow_fp8:
            raise ValueError(
                f"{field_name} cannot be {value!r}; only base projection weights "
                "support experimental fp4 storage."
            )
        return _resolve_fp4_param_dtype()
    return {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }[normalized]


def _resolve_compute_dtype(value: str, *, field_name: str):
    normalized = _canonical_dtype_name(value, field_name=field_name)
    if normalized in {"input", "fp8", "fp4"}:
        return normalized
    return {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }[normalized]


def _cast_for_compute(value, precision: ComputeDType, *, fallback_dtype):
    dtype = _resolve_compute_dtype(precision, field_name="compute_dtype")
    if hasattr(value, "dtype") and _is_low_precision_storage_dtype(value.dtype):
        if dtype not in {"input", "fp8", "fp4"}:
            return value.astype(dtype)
        return value.astype(fallback_dtype)
    if dtype in {"fp8", "fp4"}:
        return value.astype(fallback_dtype)
    if dtype == "input":
        return value
    return value.astype(dtype)


def _validate_quantized_contracting_dim(size: int, *, label: str, precision: str) -> None:
    if size % 16 != 0:
        raise ValueError(
            f"{precision.upper()} {label} contracting dimension must be a multiple of 16 for "
            f"Transformer Engine quantized GEMMs. Got {size}."
        )


def _pad_dim_to_multiple(value, axis: int, multiple: int):
    axis = axis % value.ndim
    size = value.shape[axis]
    pad = (-size) % multiple
    if pad == 0:
        return value, size
    pad_width = [(0, 0)] * value.ndim
    pad_width[axis] = (0, pad)
    return jnp.pad(value, pad_width), size


def _require_transformer_engine(precision: str):
    try:
        import transformer_engine.jax as te_jax  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional CUDA package.
        raise ImportError(
            f"{precision.upper()} LoRA requires `transformer_engine[jax]`. Install it in the active "
            "JAX environment, e.g. `pip install --no-build-isolation "
            "'transformer-engine[jax]'`."
        ) from exc
    return te_jax


def _dense(lhs, rhs, *, precision: ComputeDType, label: str):
    """Run a dense projection with selectable precision."""
    precision_name = _canonical_dtype_name(precision, field_name="precision")
    if precision_name not in {"fp8", "fp4"}:
        lhs_compute = _cast_for_compute(lhs, precision, fallback_dtype=lhs.dtype)
        rhs_compute = _cast_for_compute(rhs, precision, fallback_dtype=lhs_compute.dtype)
        return jnp.dot(lhs_compute, rhs_compute)

    lhs = _cast_for_compute(lhs, "input", fallback_dtype=lhs.dtype)
    rhs = _cast_for_compute(rhs, "input", fallback_dtype=lhs.dtype)
    if precision_name == "fp4":
        lhs = lhs.astype(jnp.bfloat16)
        rhs = rhs.astype(jnp.bfloat16)

    lhs_shape = lhs.shape
    output_size = rhs.shape[-1]
    lhs_2d = jnp.reshape(lhs, (-1, lhs_shape[-1]))
    lhs_2d, row_size = _pad_dim_to_multiple(lhs_2d, axis=0, multiple=16)
    lhs_2d, _ = _pad_dim_to_multiple(lhs_2d, axis=1, multiple=32)
    if rhs.shape[0] != lhs_2d.shape[1]:
        rhs = jnp.pad(rhs, ((0, lhs_2d.shape[1] - rhs.shape[0]), (0, 0)))
    rhs, _ = _pad_dim_to_multiple(rhs, axis=1, multiple=16)

    te_jax = _require_transformer_engine(precision_name)
    from transformer_engine.jax import dense as te_dense  # type: ignore
    from transformer_engine.jax.quantize import QuantizerFactory  # type: ignore

    autocast = getattr(te_jax, "autocast", None)
    if autocast is None:  # pragma: no cover - optional package API guard.
        raise RuntimeError(
            "Installed transformer_engine.jax does not expose autocast; "
            f"cannot enable {precision_name.upper()} LoRA."
        )

    recipe = None
    if precision_name == "fp4":
        from transformer_engine.common.recipe import NVFP4BlockScaling  # type: ignore

        recipe = NVFP4BlockScaling()

    context = autocast(enabled=True, recipe=recipe)
    with context:
        quantizer_set = QuantizerFactory.create_set()
        out = te_dense.dense(
            lhs_2d,
            rhs,
            contracting_dims=((1,), (0,)),
            quantizer_set=quantizer_set,
        )
    out = out[:row_size, :output_size]
    return jnp.reshape(out, (*lhs_shape[:-1], output_size))


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
            init=_low_precision_safe_init(w_init),
        )
        y = _dense(
            activation,
            w,
            precision=self._config.base_compute_dtype,
            label="base",
        )

        if self._with_bias:
            b_init = self._b_init or jnp.zeros
            bias_dtype = (
                "bfloat16"
                if _is_low_precision_storage_dtype(base_param_dtype)
                else self._config.base_param_dtype
            )
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
        with hk.with_rng(_adapter_parameter_rng("lora_a")):
            lora_a = hk.get_parameter(
                "lora_a",
                shape=(input_size, rank),
                dtype=lora_param_dtype,
                init=hk.initializers.RandomNormal(stddev=0.01),
            )
        with hk.with_rng(_adapter_parameter_rng("lora_b")):
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
        return y + delta.astype(y.dtype) * (self._config.alpha / self._config.rank)


def _module_path_matches(path: str, target_names: frozenset[str]) -> bool:
    normalized_path = path.replace(".", "_")
    for target in target_names:
        normalized_target = target.replace(".", "_")
        if normalized_target in normalized_path:
            return True
    return False


def _standardized_conv1d_base(
    x: Float[Array, "B S D"],
    *,
    num_channels: int,
    width: int,
) -> Float[Array, "B S O"]:
    input_channels = x.shape[-1]
    fan_in = width * input_channels
    kernel_shape = (width, input_channels, num_channels)
    w_init = hk.initializers.TruncatedNormal(stddev=1.0 / jnp.sqrt(fan_in))
    w = hk.get_parameter("w", shape=kernel_shape, dtype=x.dtype, init=w_init)

    w_standardized = w - jnp.mean(w, axis=(0, 1), keepdims=True)
    var_w = jnp.var(w_standardized, axis=(0, 1), keepdims=True)
    scale = hk.get_parameter(
        "scale",
        shape=[1, 1, num_channels],
        init=jnp.ones,
        dtype=w.dtype,
    )
    scale = scale * jax.lax.rsqrt(jnp.maximum(fan_in * var_w, 1e-4))
    w_standardized = w_standardized * scale

    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=w_standardized,
        window_strides=[1],
        padding="SAME",
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 2, 1), rhs_spec=(2, 1, 0), out_spec=(0, 2, 1)
        ),
    )
    bias = hk.get_parameter(
        "bias",
        shape=(num_channels,),
        dtype=x.dtype,
        init=hk.initializers.TruncatedNormal(stddev=1e-4),
    )
    return y + jnp.broadcast_to(bias, y.shape)


def _effective_conv1d_base(
    x: Float[Array, "B S D"],
    *,
    num_channels: int,
    width: int,
) -> Float[Array, "B S O"]:
    """Conv1D path for checkpoints that store already-standardized effective weights."""
    input_channels = x.shape[-1]
    fan_in = width * input_channels
    kernel_shape = (width, input_channels, num_channels)
    w_init = hk.initializers.TruncatedNormal(stddev=1.0 / jnp.sqrt(fan_in))
    w = hk.get_parameter("w", shape=kernel_shape, dtype=x.dtype, init=w_init)
    # Preserve the checkpoint/tree contract for former StandardizedConv1D modules.
    hk.get_parameter("scale", shape=[1, 1, num_channels], init=jnp.ones, dtype=w.dtype)

    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=w,
        window_strides=[1],
        padding="SAME",
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 2, 1), rhs_spec=(2, 1, 0), out_spec=(0, 2, 1)
        ),
    )
    bias = hk.get_parameter(
        "bias",
        shape=(num_channels,),
        dtype=x.dtype,
        init=hk.initializers.TruncatedNormal(stddev=1e-4),
    )
    return y + jnp.broadcast_to(bias, y.shape)


def _locon_delta(
    x: Float[Array, "B S D"],
    *,
    num_channels: int,
    width: int,
    config: BackboneLoConConfig,
) -> Float[Array, "B S O"]:
    input_channels = x.shape[-1]
    rank = config.rank
    if rank > num_channels:
        raise ValueError(f"LoCon rank {rank} must be <= output channels {num_channels}.")
    param_dtype = _resolve_param_dtype(
        config.param_dtype,
        field_name="locon_param_dtype",
    )
    activation = _cast_for_compute(
        x,
        config.compute_dtype,
        fallback_dtype=x.dtype,
    )
    with hk.with_rng(_adapter_parameter_rng("locon_down_w")):
        down_w = hk.get_parameter(
            "locon_down_w",
            shape=(width, input_channels, rank),
            dtype=param_dtype,
            init=hk.initializers.VarianceScaling(),
        )
    with hk.with_rng(_adapter_parameter_rng("locon_up_w")):
        up_w = hk.get_parameter(
            "locon_up_w",
            shape=(1, rank, num_channels),
            dtype=param_dtype,
            init=hk.initializers.Constant(0.0),
        )
    down_w = _cast_for_compute(
        down_w,
        config.compute_dtype,
        fallback_dtype=activation.dtype,
    )
    up_w = _cast_for_compute(
        up_w,
        config.compute_dtype,
        fallback_dtype=activation.dtype,
    )
    delta = jax.lax.conv_general_dilated(
        lhs=activation,
        rhs=down_w,
        window_strides=[1],
        padding="SAME",
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 2, 1), rhs_spec=(2, 1, 0), out_spec=(0, 2, 1)
        ),
    )
    return jax.lax.conv_general_dilated(
        lhs=delta,
        rhs=up_w,
        window_strides=[1],
        padding="SAME",
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 2, 1), rhs_spec=(2, 1, 0), out_spec=(0, 2, 1)
        ),
    )


class StandardizedConv1DWithLoCon(hk.Module):
    """Drop-in ``StandardizedConv1D`` replacement with trainable LoCon leaves."""

    def __init__(
        self,
        num_channels: int,
        width: int,
        config: BackboneLoConConfig,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        config.validate()
        self._num_channels = int(num_channels)
        self._width = int(width)
        self._config = config

    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S O"]:
        y = _standardized_conv1d_base(
            x,
            num_channels=self._num_channels,
            width=self._width,
        )
        delta = _locon_delta(
            x,
            num_channels=self._num_channels,
            width=self._width,
            config=self._config,
        )
        return y + delta.astype(y.dtype) * (self._config.alpha / self._config.rank)


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


@contextmanager
def patch_haiku_locon(config: BackboneLoConConfig) -> Iterator[None]:
    """Patch AlphaGenome ``StandardizedConv1D`` modules for selected LoCon targets."""

    from alphagenome_research.model import convolutions

    original_standardized_conv = convolutions.StandardizedConv1D
    target_names = config.normalized_target_names()

    class MaybeLoConStandardizedConv1D(hk.Module):
        def __init__(self, num_channels: int, width: int, name: str | None = None):
            super().__init__(name=name)
            self._num_channels = num_channels
            self._width = width

        def __call__(self, x):
            current_name = hk.experimental.current_name()
            y = _standardized_conv1d_base(
                x,
                num_channels=self._num_channels,
                width=self._width,
            )
            if _module_path_matches(current_name, target_names):
                delta = _locon_delta(
                    x,
                    num_channels=self._num_channels,
                    width=self._width,
                    config=config,
                )
                return y + delta.astype(y.dtype) * (config.alpha / config.rank)
            return y

    MaybeLoConStandardizedConv1D.__name__ = "StandardizedConv1D"
    MaybeLoConStandardizedConv1D.__qualname__ = "StandardizedConv1D"
    convolutions.StandardizedConv1D = MaybeLoConStandardizedConv1D
    try:
        yield
    finally:
        convolutions.StandardizedConv1D = original_standardized_conv


@contextmanager
def patch_haiku_effective_conv(effective_conv_paths: Sequence[str]) -> Iterator[None]:
    """Patch selected AlphaGenome ``StandardizedConv1D`` modules to use stored W_eff."""

    from alphagenome_research.model import convolutions

    original_standardized_conv = convolutions.StandardizedConv1D
    target_names = frozenset(str(path) for path in effective_conv_paths)

    class MaybeEffectiveStandardizedConv1D(hk.Module):
        def __init__(self, num_channels: int, width: int, name: str | None = None):
            super().__init__(name=name)
            self._num_channels = num_channels
            self._width = width

        def __call__(self, x):
            current_name = hk.experimental.current_name()
            if _module_path_matches(current_name, target_names):
                return _effective_conv1d_base(
                    x,
                    num_channels=self._num_channels,
                    width=self._width,
                )
            return _standardized_conv1d_base(
                x,
                num_channels=self._num_channels,
                width=self._width,
            )

    MaybeEffectiveStandardizedConv1D.__name__ = "StandardizedConv1D"
    MaybeEffectiveStandardizedConv1D.__qualname__ = "StandardizedConv1D"
    convolutions.StandardizedConv1D = MaybeEffectiveStandardizedConv1D
    try:
        yield
    finally:
        convolutions.StandardizedConv1D = original_standardized_conv


@contextmanager
def patch_backbone_adapters(
    lora_config: BackboneLoRAConfig | None = None,
    locon_config: BackboneLoConConfig | None = None,
    effective_conv_paths: Sequence[str] | None = None,
) -> Iterator[None]:
    """Patch all requested backbone adapter module types in one scoped context."""

    if lora_config is None and locon_config is None and not effective_conv_paths:
        yield
    else:
        with ExitStack() as stack:
            if effective_conv_paths:
                stack.enter_context(patch_haiku_effective_conv(effective_conv_paths))
            if lora_config is not None:
                stack.enter_context(patch_haiku_linear(lora_config))
            if locon_config is not None:
                stack.enter_context(patch_haiku_locon(locon_config))
            yield


def parse_lora_target_names(raw: str | None) -> tuple[str, ...]:
    """Parse comma-separated target names from CLI/env settings."""
    if raw is None or raw.strip().lower() in {"", "default", "all"}:
        return DEFAULT_BACKBONE_LORA_TARGETS
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def parse_locon_target_names(raw: str | None) -> tuple[str, ...]:
    """Parse comma-separated LoCon target path substrings from CLI/env settings."""
    if raw is None or raw.strip().lower() in {"", "default", "all"}:
        return DEFAULT_BACKBONE_LOCON_TARGETS
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def expand_adapter_parameter_tree(
    source,
    target,
    *,
    source_lora_config: BackboneLoRAConfig | None,
    target_lora_config: BackboneLoRAConfig | None,
    source_locon_config: BackboneLoConConfig | None,
    target_locon_config: BackboneLoConConfig | None,
):
    """Transplant a checkpoint into a model with expanded LoRA or LoCon adapters.

    Existing low-rank directions occupy the leading rank dimensions. Newly added
    directions retain the target model's deterministic down-projection and zero
    up-projection initialization. Up projections are rescaled when ``alpha / rank``
    changes, preserving the source model's effective adapter residual exactly.
    """

    stats = {
        "copied_leaves": 0,
        "expanded_leaves": 0,
        "initialized_adapter_leaves": 0,
    }
    adapter_leaf_names = {"lora_a", "lora_b", "locon_down_w", "locon_up_w"}

    def count_initialized_adapters(node) -> int:
        if not isinstance(node, Mapping):
            return 0
        return sum(
            1 if str(key) in adapter_leaf_names else count_initialized_adapters(value)
            for key, value in node.items()
        )

    def adapter_scale(config, *, kind: str) -> float:
        if config is None:
            raise ValueError(f"Cannot transplant a source {kind} leaf without its config.")
        return float(config.alpha) / int(config.rank)

    def transplant_leaf(path: tuple[str, ...], source_leaf, target_leaf):
        name = path[-1]
        source_shape = tuple(source_leaf.shape)
        target_shape = tuple(target_leaf.shape)

        if name == "lora_a":
            if source_shape[:-1] != target_shape[:-1] or source_shape[-1] > target_shape[-1]:
                raise ValueError(f"Cannot expand {'/'.join(path)} from {source_shape} to {target_shape}.")
            result = target_leaf.at[..., : source_shape[-1]].set(source_leaf.astype(target_leaf.dtype))
        elif name == "lora_b":
            if source_shape[1:] != target_shape[1:] or source_shape[0] > target_shape[0]:
                raise ValueError(f"Cannot expand {'/'.join(path)} from {source_shape} to {target_shape}.")
            ratio = adapter_scale(source_lora_config, kind="LoRA") / adapter_scale(
                target_lora_config, kind="LoRA"
            )
            result = target_leaf.at[: source_shape[0], ...].set(
                source_leaf.astype(target_leaf.dtype) * ratio
            )
        elif name == "locon_down_w":
            if source_shape[:-1] != target_shape[:-1] or source_shape[-1] > target_shape[-1]:
                raise ValueError(f"Cannot expand {'/'.join(path)} from {source_shape} to {target_shape}.")
            result = target_leaf.at[..., : source_shape[-1]].set(source_leaf.astype(target_leaf.dtype))
        elif name == "locon_up_w":
            if (
                source_shape[0] != target_shape[0]
                or source_shape[2:] != target_shape[2:]
                or source_shape[1] > target_shape[1]
            ):
                raise ValueError(f"Cannot expand {'/'.join(path)} from {source_shape} to {target_shape}.")
            ratio = adapter_scale(source_locon_config, kind="LoCon") / adapter_scale(
                target_locon_config, kind="LoCon"
            )
            result = target_leaf.at[:, : source_shape[1], ...].set(
                source_leaf.astype(target_leaf.dtype) * ratio
            )
        elif source_shape == target_shape:
            stats["copied_leaves"] += 1
            return source_leaf.astype(target_leaf.dtype)
        else:
            raise ValueError(
                f"Non-adapter leaf {'/'.join(path)} changed shape from "
                f"{source_shape} to {target_shape}."
            )

        if source_shape == target_shape:
            stats["copied_leaves"] += 1
        else:
            stats["expanded_leaves"] += 1
        return result

    def visit(source_node, target_node, path: tuple[str, ...]):
        if isinstance(target_node, Mapping):
            if not isinstance(source_node, Mapping):
                raise ValueError(f"Checkpoint tree mismatch at {'/'.join(path)}.")
            missing_targets = set(source_node) - set(target_node)
            if missing_targets:
                raise ValueError(
                    f"Expanded model dropped source keys at {'/'.join(path)}, "
                    f"{sorted(str(key) for key in missing_targets)}."
                )
            result = {}
            for key, target_value in target_node.items():
                key_path = (*path, str(key))
                if key in source_node:
                    result[key] = visit(source_node[key], target_value, key_path)
                else:
                    result[key] = target_value
                    stats["initialized_adapter_leaves"] += (
                        1
                        if str(key) in adapter_leaf_names
                        else count_initialized_adapters(target_value)
                    )
            return result
        if isinstance(source_node, Mapping):
            raise ValueError(f"Checkpoint tree mismatch at {'/'.join(path)}.")
        return transplant_leaf(path, source_node, target_node)

    return visit(source, target, ()), stats


__all__ = [
    "BackboneLoConConfig",
    "BackboneLoRAConfig",
    "DEFAULT_BACKBONE_LOCON_TARGETS",
    "DEFAULT_BACKBONE_LORA_TARGETS",
    "LinearWithLoRA",
    "StandardizedConv1DWithLoCon",
    "expand_adapter_parameter_tree",
    "parse_locon_target_names",
    "parse_lora_target_names",
    "patch_backbone_adapters",
    "patch_haiku_effective_conv",
    "patch_haiku_linear",
    "patch_haiku_locon",
]
