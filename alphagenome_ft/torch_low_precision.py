"""Local PyTorch low-precision conversion helpers for inference ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Float8ConversionStats:
    """Summary of a torchao float8 conversion pass."""

    backend: str
    recipe: str
    converted_linears: int
    skipped_linears: int
    min_feature_multiple: int
    skipped_name_patterns: tuple[str, ...]
    include_name_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Float4ConversionStats:
    """Summary of a 4-bit linear conversion pass."""

    backend: str
    mode: str
    converted_linears: int
    skipped_linears: int
    min_feature_multiple: int
    skipped_name_patterns: tuple[str, ...]
    include_name_patterns: tuple[str, ...] = ()
    quant_type: str | None = None


def _matches_any_pattern(name: str, patterns: Iterable[str]) -> bool:
    return any(pattern and pattern in name for pattern in patterns)


def _matches_include_patterns(name: str, patterns: Iterable[str]) -> bool:
    patterns = tuple(patterns)
    return not patterns or _matches_any_pattern(name, patterns)


class _ContiguousInputWrapper(nn.Module):
    """Wrap prototype quantized modules that require contiguous inputs."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        if x.dim() <= 2:
            return self.module(x)
        leading_shape = x.shape[:-1]
        y = self.module(x.view(-1, x.shape[-1]))
        return y.view(*leading_shape, y.shape[-1])


class PointwiseConv1dAsLinear(nn.Module):
    """Run a Conv1d(k=1) as a position-wise Linear on NCL tensors."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        if conv.kernel_size != (1,) or conv.stride != (1,) or conv.dilation != (1,) or conv.groups != 1:
            raise ValueError("PointwiseConv1dAsLinear only supports ungrouped Conv1d(k=1).")
        if conv.padding not in {(0,), "valid"}:
            raise ValueError("PointwiseConv1dAsLinear only supports unpadded Conv1d(k=1).")

        self.linear = nn.Linear(
            conv.in_channels,
            conv.out_channels,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            self.linear.weight.copy_(conv.weight.squeeze(-1))
            if conv.bias is not None:
                self.linear.bias.copy_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape
        y = x.transpose(1, 2).contiguous().view(-1, leading_shape[1])
        y = self.linear(y)
        return y.view(leading_shape[0], leading_shape[2], -1).transpose(1, 2).contiguous()


def _wrap_linears_by_weight_class_name(model: nn.Module, class_name: str) -> int:
    named_modules = dict(model.named_modules())
    wrapped = 0
    for fqn, module in list(named_modules.items()):
        if fqn == "" or not isinstance(module, nn.Linear):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.__class__.__name__ != class_name:
            continue
        child_name = fqn.split(".")[-1]
        parent_fqn = fqn.removesuffix(child_name).removesuffix(".")
        parent = named_modules[parent_fqn]
        setattr(parent, child_name, _ContiguousInputWrapper(module))
        wrapped += 1
    return wrapped


def convert_pointwise_conv1d_to_linear(
    model: nn.Module,
    *,
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = ("heads", "lora_", "locon_", "ia3", "adapter"),
    include_name_patterns: Iterable[str] = (),
) -> dict[str, object]:
    """Replace eligible Conv1d(k=1) modules with Linear wrappers.

    This enables true Linear quantization backends to cover pointwise Conv1d
    projections without changing numerics beyond normal operator ordering.
    Wider Conv1d kernels are intentionally left untouched because these
    backends do not provide a CUDA Conv1d weight-only module here.
    """
    if min_feature_multiple < 1:
        raise ValueError("min_feature_multiple must be >= 1")

    patterns = tuple(skip_name_patterns)
    include_patterns = tuple(include_name_patterns)
    named_modules = dict(model.named_modules())
    decisions: dict[str, bool] = {}
    replacements: list[tuple[str, nn.Module]] = []

    for fqn, module in list(named_modules.items()):
        if fqn == "" or not isinstance(module, nn.Conv1d):
            continue
        eligible = (
            module.kernel_size == (1,)
            and module.stride == (1,)
            and module.dilation == (1,)
            and module.groups == 1
            and module.in_channels % min_feature_multiple == 0
            and module.out_channels % min_feature_multiple == 0
            and _matches_include_patterns(fqn, include_patterns)
            and not _matches_any_pattern(fqn, patterns)
        )
        decisions[fqn] = eligible
        if eligible:
            replacements.append((fqn, PointwiseConv1dAsLinear(module)))

    for fqn, replacement in replacements:
        child_name = fqn.split(".")[-1]
        parent_fqn = fqn.removesuffix(child_name).removesuffix(".")
        parent = named_modules[parent_fqn]
        setattr(parent, child_name, replacement)

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return {
        "converted_pointwise_convs": converted,
        "skipped_pointwise_convs": skipped,
        "pointwise_conv_skip_name_patterns": patterns,
        "pointwise_conv_include_name_patterns": include_patterns,
    }


def convert_linears_to_float8_training(
    model: nn.Module,
    *,
    recipe: str = "tensorwise",
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = (
        "heads",
        "original_layer",
        "lora_",
        "locon_",
        "ia3",
        "adapter",
    ),
    include_name_patterns: Iterable[str] = (),
) -> Float8ConversionStats:
    """Convert eligible ``nn.Linear`` modules to torchao Float8Linear."""
    if recipe not in {"tensorwise", "rowwise", "rowwise_with_gw_hp"}:
        raise ValueError(
            "Unsupported float8 recipe "
            f"{recipe!r}; expected tensorwise, rowwise, or rowwise_with_gw_hp"
        )
    if min_feature_multiple < 1:
        raise ValueError("min_feature_multiple must be >= 1")

    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:
        raise RuntimeError("torchao is required for float8 linear conversion.") from exc

    patterns = tuple(skip_name_patterns)
    include_patterns = tuple(include_name_patterns)
    decisions: dict[str, bool] = {}

    def module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        eligible = (
            isinstance(mod, nn.Linear)
            and mod.in_features % min_feature_multiple == 0
            and mod.out_features % min_feature_multiple == 0
            and _matches_include_patterns(fqn, include_patterns)
            and not _matches_any_pattern(fqn, patterns)
        )
        if isinstance(mod, nn.Linear):
            decisions[fqn] = eligible
        return eligible

    config = Float8LinearConfig.from_recipe_name(recipe)
    convert_to_float8_training(model, config=config, module_filter_fn=module_filter_fn)

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return Float8ConversionStats(
        backend="torchao",
        recipe=recipe,
        converted_linears=converted,
        skipped_linears=skipped,
        min_feature_multiple=min_feature_multiple,
        skipped_name_patterns=patterns,
        include_name_patterns=include_patterns,
    )


def convert_linears_to_nvfp4_weight_only(
    model: nn.Module,
    *,
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = ("heads", "lora_", "locon_", "ia3", "adapter"),
    include_name_patterns: Iterable[str] = (),
    use_dynamic_per_tensor_scale: bool = True,
) -> Float4ConversionStats:
    """Convert eligible frozen ``nn.Linear`` weights to torchao NVFP4 tensors."""
    if min_feature_multiple < 1:
        raise ValueError("min_feature_multiple must be >= 1")

    try:
        from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError("torchao is required for NVFP4 weight-only conversion.") from exc

    patterns = tuple(skip_name_patterns)
    include_patterns = tuple(include_name_patterns)
    decisions: dict[str, bool] = {}

    def module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        eligible = (
            isinstance(mod, nn.Linear)
            and mod.in_features % min_feature_multiple == 0
            and mod.out_features % min_feature_multiple == 0
            and not mod.weight.requires_grad
            and _matches_include_patterns(fqn, include_patterns)
            and not _matches_any_pattern(fqn, patterns)
        )
        if isinstance(mod, nn.Linear):
            decisions[fqn] = eligible
        return eligible

    quantize_(
        model,
        NVFP4WeightOnlyConfig(use_dynamic_per_tensor_scale=use_dynamic_per_tensor_scale),
        filter_fn=module_filter_fn,
    )
    _wrap_linears_by_weight_class_name(model, "NVFP4Tensor")

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return Float4ConversionStats(
        backend="torchao",
        mode="weight_only",
        converted_linears=converted,
        skipped_linears=skipped,
        min_feature_multiple=min_feature_multiple,
        skipped_name_patterns=patterns,
        include_name_patterns=include_patterns,
        quant_type="nvfp4",
    )


def convert_linears_to_bnb_nf4_weight_only(
    model: nn.Module,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = ("heads", "lora_", "locon_", "ia3", "adapter"),
    include_name_patterns: Iterable[str] = (),
) -> Float4ConversionStats:
    """Replace eligible frozen ``nn.Linear`` modules with bitsandbytes NF4 linears."""
    if min_feature_multiple < 1:
        raise ValueError("min_feature_multiple must be >= 1")
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("bitsandbytes is required for NF4 linear conversion.") from exc

    patterns = tuple(skip_name_patterns)
    include_patterns = tuple(include_name_patterns)
    named_modules = dict(model.named_modules())
    decisions: dict[str, bool] = {}
    replacements: list[tuple[str, nn.Module]] = []

    for fqn, module in list(named_modules.items()):
        if fqn == "" or not isinstance(module, nn.Linear):
            continue
        eligible = (
            module.in_features % min_feature_multiple == 0
            and module.out_features % min_feature_multiple == 0
            and not module.weight.requires_grad
            and _matches_include_patterns(fqn, include_patterns)
            and not _matches_any_pattern(fqn, patterns)
        )
        decisions[fqn] = eligible
        if not eligible:
            continue
        quantized = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            quant_type="nf4",
            quant_storage=torch.uint8,
        )
        quantized.load_state_dict(module.state_dict(), strict=True)
        quantized.train(module.training)
        quantized = quantized.to(device=module.weight.device)
        for param in quantized.parameters():
            param.requires_grad_(False)
        replacements.append((fqn, quantized))

    for fqn, quantized in replacements:
        child_name = fqn.split(".")[-1]
        parent_fqn = fqn.removesuffix(child_name).removesuffix(".")
        parent = named_modules[parent_fqn]
        setattr(parent, child_name, quantized)

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return Float4ConversionStats(
        backend="bitsandbytes",
        mode="weight_only",
        converted_linears=converted,
        skipped_linears=skipped,
        min_feature_multiple=min_feature_multiple,
        skipped_name_patterns=patterns,
        include_name_patterns=include_patterns,
        quant_type="nf4",
    )
