"""Torch helpers for JAX checkpoints that store effective conv weights."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class EffectiveConv1d(nn.Conv1d):
    """Conv1d with AlphaGenome same-padding but no weight standardization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: str = "same",
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.pad_mode = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_mode == "same":
            pad_total = self.kernel_size[0] - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right))
        return F.conv1d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


def jax_effective_path_to_torch(path: str) -> str:
    """Map a JAX StandardizedConv1D module path to its Torch module path."""
    normalized = path.strip("/")
    match = re.fullmatch(
        r"alphagenome/sequence_encoder/downres_block_(\d+)/(conv_block(?:_1)?)/standardized_conv1_d",
        normalized,
    )
    if match is None:
        raise ValueError(f"Unsupported effective conv JAX path: {path!r}")
    block_idx = int(match.group(1))
    block_name = "block2" if match.group(2) == "conv_block_1" else "block1"
    return f"encoder.down_blocks.{block_idx}.{block_name}.conv"


def jax_effective_paths_to_torch(paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(jax_effective_path_to_torch(path) for path in paths)


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def materialize_effective_convs(model: nn.Module, module_names: Sequence[str]) -> tuple[str, ...]:
    """Replace selected standardized conv modules with direct effective conv modules."""
    converted: list[str] = []
    for name in module_names:
        module = model.get_submodule(name)
        replacement = EffectiveConv1d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=getattr(module, "pad_mode", "same"),
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
        )
        replacement.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(module.weight)
            if module.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(module.bias)
        _set_submodule(model, name, replacement)
        converted.append(name)
    return tuple(converted)


def _standardized_weight(module: nn.Conv1d) -> torch.Tensor:
    weight = module.weight
    mean = weight.mean(dim=(1, 2), keepdim=True)
    var = weight.var(dim=(1, 2), keepdim=True, unbiased=False)
    fan_in = module.in_channels * module.kernel_size[0]
    floor = torch.tensor(1e-4, device=weight.device, dtype=weight.dtype)
    scale = torch.rsqrt(torch.maximum(var * fan_in, floor))
    learned_scale = getattr(module, "scale", None)
    if learned_scale is not None:
        scale = scale * learned_scale
    return (weight - mean) * scale


def materialize_standardized_convs(model: nn.Module) -> tuple[str, ...]:
    """Precompute StandardizedConv1d weights and replace them with direct convs."""
    converted: list[str] = []
    for name, module in list(model.named_modules()):
        if name == "" or module.__class__.__name__ != "StandardizedConv1d":
            continue
        if not isinstance(module, nn.Conv1d):
            continue
        replacement = EffectiveConv1d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=getattr(module, "pad_mode", "same"),
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
        )
        replacement.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(_standardized_weight(module))
            if module.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(module.bias)
        _set_submodule(model, name, replacement)
        converted.append(name)
    return tuple(converted)
