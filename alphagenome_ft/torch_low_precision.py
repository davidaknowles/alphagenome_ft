"""Local PyTorch low-precision conversion helpers for inference ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA path
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _maxpool1d_k2s2_no_indices_kernel(
        x_ptr,
        out_ptr,
        total_outputs: tl.constexpr,
        channels: tl.constexpr,
        length: tl.constexpr,
        out_length: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        pos = offsets % out_length
        channel_idx = (offsets // out_length) % channels
        batch_idx = offsets // (channels * out_length)
        input0 = (
            batch_idx.to(tl.int64) * channels * length
            + channel_idx.to(tl.int64) * length
            + (pos * 2).to(tl.int64)
        )
        input1 = input0 + 1
        mask = offsets < total_outputs
        x0 = tl.load(x_ptr + input0, mask=mask, other=-float("inf")).to(tl.float32)
        x1 = tl.load(x_ptr + input1, mask=mask & ((pos * 2 + 1) < length), other=-float("inf")).to(tl.float32)
        tl.store(out_ptr + offsets, tl.maximum(x0, x1), mask=mask)


    @triton.jit
    def _norm_gelu_int8_weight_only_conv1d_kernel(
        x_ptr,
        inv_ptr,
        norm_bias_ptr,
        qweight_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        total_positions: tl.constexpr,
        in_channels: tl.constexpr,
        out_channels: tl.constexpr,
        length: tl.constexpr,
        kernel_width: tl.constexpr,
        pad_left: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        batch_idx = offs_m // length
        pos_idx = offs_m - batch_idx * length

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_start in range(0, in_channels * kernel_width, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            chan_idx = offs_k // kernel_width
            kernel_idx = offs_k - chan_idx * kernel_width
            input_pos = pos_idx[:, None] + kernel_idx[None, :] - pad_left
            x_offsets = (
                batch_idx[:, None].to(tl.int64) * in_channels * length
                + chan_idx[None, :].to(tl.int64) * length
                + input_pos.to(tl.int64)
            )
            x_mask = (
                (offs_m[:, None] < total_positions)
                & (offs_k[None, :] < in_channels * kernel_width)
                & (input_pos >= 0)
                & (input_pos < length)
            )
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
            inv = tl.load(inv_ptr + chan_idx, mask=offs_k < in_channels * kernel_width, other=0.0).to(tl.float32)
            nbias = tl.load(norm_bias_ptr + chan_idx, mask=offs_k < in_channels * kernel_width, other=0.0).to(tl.float32)
            h = x_vals * inv[None, :] + nbias[None, :]
            h = h * tl.sigmoid(1.702 * h)
            h = tl.where(x_mask, h, 0.0)
            w_offsets = offs_n[None, :] * in_channels * kernel_width + offs_k[:, None]
            w_mask = (offs_n[None, :] < out_channels) & (offs_k[:, None] < in_channels * kernel_width)
            w_vals = tl.load(qweight_ptr + w_offsets, mask=w_mask, other=0).to(tl.float32)
            acc += tl.dot(h, w_vals, input_precision="tf32")

        scales = tl.load(scale_ptr + offs_n, mask=offs_n < out_channels, other=0.0).to(tl.float32)
        acc = acc * scales[None, :]
        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_channels, other=0.0).to(tl.float32)
            acc += bias[None, :]
        out_offsets = (
            batch_idx[:, None].to(tl.int64) * out_channels * length
            + offs_n[None, :].to(tl.int64) * length
            + pos_idx[:, None].to(tl.int64)
        )
        out_mask = (offs_m[:, None] < total_positions) & (offs_n[None, :] < out_channels)
        tl.store(out_ptr + out_offsets, acc, mask=out_mask)


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


class TritonMaxPool1dK2S2NoIndices(nn.Module):
    """Triton max-pool for encoder k=2/s=2 without materializing indices."""

    def __init__(self, pool: nn.Module):
        super().__init__()
        if triton is None:
            raise RuntimeError("triton is required for TritonMaxPool1dK2S2NoIndices.")
        if getattr(pool, "kernel_size", None) != 2 or getattr(pool, "stride", None) != 2:
            raise ValueError("TritonMaxPool1dK2S2NoIndices only supports kernel_size=stride=2.")
        if getattr(pool, "method", "max") != "max":
            raise ValueError("TritonMaxPool1dK2S2NoIndices only supports max pooling.")
        self.kernel_size = 2
        self.stride = 2
        self.method = "max"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected NCL input, got shape {tuple(x.shape)}")
        if not x.is_cuda:
            return F.max_pool1d(x, kernel_size=2, stride=2)
        x = x.contiguous()
        batch, channels, length = x.shape
        out_length = (length + 1) // 2
        out = torch.empty((batch, channels, out_length), device=x.device, dtype=x.dtype)
        total_outputs = batch * channels * out_length
        grid = (triton.cdiv(total_outputs, 256),)
        _maxpool1d_k2s2_no_indices_kernel[grid](
            x,
            out,
            total_outputs,
            channels,
            length,
            out_length,
            BLOCK=256,
        )
        return out


class FusedNormGeluInt8WeightOnlyConv1d(nn.Module):
    """Fuse RMSBatchNorm, gelu, and int8 weight-only Conv1d for one ConvBlock."""

    def __init__(self, block: nn.Module):
        super().__init__()
        if triton is None:
            raise RuntimeError("triton is required for FusedNormGeluInt8WeightOnlyConv1d.")
        norm = block.norm
        conv = block.conv
        if not isinstance(conv, nn.Conv1d):
            raise ValueError("FusedNormGeluInt8WeightOnlyConv1d expects a ConvBlock with Conv1d.")
        if conv.stride != (1,) or conv.dilation != (1,) or conv.groups != 1:
            raise ValueError("FusedNormGeluInt8WeightOnlyConv1d only supports stride=1, dilation=1, groups=1.")
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.pad_mode = getattr(conv, "pad_mode", conv.padding)
        if self.pad_mode == "same":
            self.pad_left = (self.kernel_size[0] - 1) // 2
        elif self.pad_mode in {(0,), 0, "valid"}:
            self.pad_left = 0
        else:
            raise ValueError(f"Unsupported fused ConvBlock padding mode: {self.pad_mode!r}")

        inv = norm.weight.detach().float() * torch.rsqrt(norm.running_var.detach().float() + norm.eps)
        self.register_buffer("norm_inv", inv.to(device=conv.weight.device, dtype=torch.float32).contiguous())
        self.register_buffer("norm_bias", norm.bias.detach().float().to(device=conv.weight.device).contiguous())

        weight = conv.weight.detach().float()
        flat = weight.reshape(weight.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        qweight = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.to(device=conv.weight.device, dtype=torch.float32))
        if conv.bias is None:
            self.register_buffer("bias", torch.empty(0, device=conv.weight.device, dtype=torch.float32))
            self.has_bias = False
        else:
            self.register_buffer("bias", conv.bias.detach().float().contiguous())
            self.has_bias = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected NCL input, got shape {tuple(x.shape)}")
        if not x.is_cuda:
            h = (
                x * self.norm_inv.to(x.device, dtype=x.dtype).view(1, -1, 1)
                + self.norm_bias.to(x.device, dtype=x.dtype).view(1, -1, 1)
            )
            h = h * torch.sigmoid(torch.tensor(1.702, dtype=h.dtype, device=h.device) * h)
            weight = (self.qweight.float() * self.scale[:, None]).reshape(
                self.out_channels, self.in_channels, self.kernel_size[0]
            )
            bias = self.bias.to(x.device, dtype=x.dtype) if self.has_bias else None
            if self.pad_mode == "same":
                pad_total = self.kernel_size[0] - 1
                h = F.pad(h, (pad_total // 2, pad_total - pad_total // 2))
            return F.conv1d(h, weight.to(x.device, dtype=x.dtype), bias)
        x = x.contiguous()
        batch, in_channels, length = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}.")
        out = torch.empty((batch, self.out_channels, length), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(batch * length, 32), triton.cdiv(self.out_channels, 32))
        _norm_gelu_int8_weight_only_conv1d_kernel[grid](
            x,
            self.norm_inv,
            self.norm_bias,
            self.qweight,
            self.scale,
            self.bias,
            out,
            batch * length,
            self.in_channels,
            self.out_channels,
            length,
            self.kernel_size[0],
            self.pad_left,
            self.has_bias,
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=64,
        )
        return out


class FusedDownResBlock0(nn.Module):
    """Memory-lean down block wrapper for inference at the first encoder scale."""

    def __init__(self, block: nn.Module):
        super().__init__()
        if triton is None:
            raise RuntimeError("triton is required for FusedDownResBlock0.")
        if not hasattr(block, "block1") or not hasattr(block, "block2"):
            raise ValueError("FusedDownResBlock0 expects a DownResBlock-like module.")
        block1 = block.block1
        block2 = block.block2
        in_channels = int(getattr(block1, "in_channels", 0))
        out_channels = int(getattr(block1, "out_channels", 0))
        if in_channels <= 0 or out_channels <= in_channels:
            raise ValueError("FusedDownResBlock0 expects block1 to increase channels.")
        if int(getattr(block2, "in_channels", 0)) != out_channels:
            raise ValueError("FusedDownResBlock0 expects block2 input channels to match block1 output.")
        if int(getattr(block2, "out_channels", 0)) != out_channels:
            raise ValueError("FusedDownResBlock0 expects block2 to preserve channels.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = FusedNormGeluInt8WeightOnlyConv1d(block1)
        self.block2 = FusedNormGeluInt8WeightOnlyConv1d(block2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        if torch.is_grad_enabled() and x.requires_grad:
            x_padded = F.pad(x, (0, 0, 0, self.out_channels - self.in_channels))
            out = out + x_padded
            return out + self.block2(out)

        out[:, : self.in_channels, :].add_(x)
        residual = out
        out = self.block2(residual)
        out.add_(residual)
        return out


if triton is not None:

    @triton.jit
    def _int8_weight_only_conv1d_kernel(
        x_ptr,
        qweight_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        total_positions: tl.constexpr,
        batch: tl.constexpr,
        in_channels: tl.constexpr,
        out_channels: tl.constexpr,
        length: tl.constexpr,
        kernel_width: tl.constexpr,
        pad_left: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        batch_idx = offs_m // length
        pos_idx = offs_m - batch_idx * length

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_start in range(0, in_channels * kernel_width, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            chan_idx = offs_k // kernel_width
            kernel_idx = offs_k - chan_idx * kernel_width
            input_pos = pos_idx[:, None] + kernel_idx[None, :] - pad_left
            # Long 131k windows at larger batch sizes exceed int32 offsets.
            x_offsets = (
                batch_idx[:, None].to(tl.int64) * in_channels * length
                + chan_idx[None, :].to(tl.int64) * length
                + input_pos.to(tl.int64)
            )
            x_mask = (
                (offs_m[:, None] < total_positions)
                & (offs_k[None, :] < in_channels * kernel_width)
                & (input_pos >= 0)
                & (input_pos < length)
            )
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
            w_offsets = offs_n[None, :] * in_channels * kernel_width + offs_k[:, None]
            w_mask = (offs_n[None, :] < out_channels) & (offs_k[:, None] < in_channels * kernel_width)
            w_vals = tl.load(qweight_ptr + w_offsets, mask=w_mask, other=0).to(tl.float32)
            acc += tl.dot(x_vals, w_vals, input_precision="tf32")

        scales = tl.load(scale_ptr + offs_n, mask=offs_n < out_channels, other=0.0).to(tl.float32)
        acc = acc * scales[None, :]
        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_channels, other=0.0).to(tl.float32)
            acc += bias[None, :]
        out_offsets = (
            batch_idx[:, None].to(tl.int64) * out_channels * length
            + offs_n[None, :].to(tl.int64) * length
            + pos_idx[:, None].to(tl.int64)
        )
        out_mask = (offs_m[:, None] < total_positions) & (offs_n[None, :] < out_channels)
        tl.store(out_ptr + out_offsets, acc, mask=out_mask)


    @triton.jit
    def _int8_dynamic_activation_int8_weight_conv1d_kernel(
        x_ptr,
        qweight_ptr,
        weight_scale_ptr,
        bias_ptr,
        out_ptr,
        total_positions: tl.constexpr,
        batch: tl.constexpr,
        in_channels: tl.constexpr,
        out_channels: tl.constexpr,
        length: tl.constexpr,
        kernel_width: tl.constexpr,
        pad_left: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        batch_idx = offs_m // length
        pos_idx = offs_m - batch_idx * length
        weight_scale = tl.load(
            weight_scale_ptr + offs_n, mask=offs_n < out_channels, other=0.0
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_start in range(0, in_channels * kernel_width, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            chan_idx = offs_k // kernel_width
            kernel_idx = offs_k - chan_idx * kernel_width
            input_pos = pos_idx[:, None] + kernel_idx[None, :] - pad_left
            # Long 131k windows at larger batch sizes exceed int32 offsets.
            x_offsets = (
                batch_idx[:, None].to(tl.int64) * in_channels * length
                + chan_idx[None, :].to(tl.int64) * length
                + input_pos.to(tl.int64)
            )
            x_mask = (
                (offs_m[:, None] < total_positions)
                & (offs_k[None, :] < in_channels * kernel_width)
                & (input_pos >= 0)
                & (input_pos < length)
            )
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
            x_absmax = tl.max(tl.abs(x_vals), axis=1)
            x_scale = tl.maximum(x_absmax / 127.0, 1.0e-8)
            x_quant = tl.extra.libdevice.nearbyint(x_vals / x_scale[:, None])
            x_quant = tl.minimum(tl.maximum(x_quant, -127.0), 127.0).to(tl.int8)

            w_offsets = offs_n[None, :] * in_channels * kernel_width + offs_k[:, None]
            w_mask = (offs_n[None, :] < out_channels) & (offs_k[:, None] < in_channels * kernel_width)
            w_vals = tl.load(qweight_ptr + w_offsets, mask=w_mask, other=0)
            dot = tl.dot(x_quant, w_vals, out_dtype=tl.int32)
            acc += dot.to(tl.float32) * x_scale[:, None] * weight_scale[None, :]

        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_channels, other=0.0).to(tl.float32)
            acc += bias[None, :]
        out_offsets = (
            batch_idx[:, None].to(tl.int64) * out_channels * length
            + offs_n[None, :].to(tl.int64) * length
            + pos_idx[:, None].to(tl.int64)
        )
        out_mask = (offs_m[:, None] < total_positions) & (offs_n[None, :] < out_channels)
        tl.store(out_ptr + out_offsets, acc, mask=out_mask)


class Int8WeightOnlyConv1d(nn.Module):
    """Triton int8 weight-only Conv1d for NCL inference tensors."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        if triton is None:
            raise RuntimeError("triton is required for Int8WeightOnlyConv1d.")
        if conv.stride != (1,) or conv.dilation != (1,) or conv.groups != 1:
            raise ValueError("Int8WeightOnlyConv1d only supports stride=1, dilation=1, groups=1.")
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.pad_mode = getattr(conv, "pad_mode", conv.padding)
        if self.pad_mode == "same":
            self.pad_left = (self.kernel_size[0] - 1) // 2
        elif self.pad_mode in {(0,), 0, "valid"}:
            self.pad_left = 0
        else:
            raise ValueError(f"Unsupported Int8WeightOnlyConv1d padding mode: {self.pad_mode!r}")

        weight = conv.weight.detach().float()
        flat = weight.reshape(weight.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        qweight = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.to(device=conv.weight.device, dtype=torch.float32))
        if conv.bias is None:
            self.register_buffer("bias", torch.empty(0, device=conv.weight.device, dtype=torch.float32))
            self.has_bias = False
        else:
            self.register_buffer("bias", conv.bias.detach().float().contiguous())
            self.has_bias = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected NCL input, got shape {tuple(x.shape)}")
        if not x.is_cuda:
            weight = (self.qweight.float() * self.scale[:, None]).reshape(
                self.out_channels, self.in_channels, self.kernel_size[0]
            )
            bias = self.bias.to(x.device, dtype=x.dtype) if self.has_bias else None
            if self.pad_mode == "same":
                pad_total = self.kernel_size[0] - 1
                x = F.pad(x, (pad_total // 2, pad_total - pad_total // 2))
            return F.conv1d(x, weight.to(x.device, dtype=x.dtype), bias)
        x = x.contiguous()
        batch, in_channels, length = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}.")
        out = torch.empty((batch, self.out_channels, length), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(batch * length, 32), triton.cdiv(self.out_channels, 32))
        _int8_weight_only_conv1d_kernel[grid](
            x,
            self.qweight,
            self.scale,
            self.bias,
            out,
            batch * length,
            batch,
            self.in_channels,
            self.out_channels,
            length,
            self.kernel_size[0],
            self.pad_left,
            self.has_bias,
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=64,
        )
        return out


class Int8DynamicActivationInt8WeightConv1d(Int8WeightOnlyConv1d):
    """Triton dynamic-int8 activation and int8 weight Conv1d."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected NCL input, got shape {tuple(x.shape)}")
        if not x.is_cuda:
            return super().forward(x)
        x = x.contiguous()
        batch, in_channels, length = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}.")
        out = torch.empty((batch, self.out_channels, length), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(batch * length, 32), triton.cdiv(self.out_channels, 32))
        _int8_dynamic_activation_int8_weight_conv1d_kernel[grid](
            x,
            self.qweight,
            self.scale,
            self.bias,
            out,
            batch * length,
            batch,
            self.in_channels,
            self.out_channels,
            length,
            self.kernel_size[0],
            self.pad_left,
            self.has_bias,
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=64,
        )
        return out


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


def _iter_named_child_modules(model: nn.Module):
    named_modules = dict(model.named_modules())
    for fqn, module in list(named_modules.items()):
        if fqn == "":
            continue
        child_name = fqn.split(".")[-1]
        parent_fqn = fqn.removesuffix(child_name).removesuffix(".")
        yield fqn, module, named_modules[parent_fqn], child_name


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


def replace_encoder_pool_with_triton_no_indices(model: nn.Module) -> dict[str, object]:
    """Replace the shared encoder max-pool with a Triton no-indices k=2/s=2 pool."""
    if triton is None:
        raise RuntimeError("triton is required for Triton encoder pool replacement.")
    pool = model.encoder.pool
    if isinstance(pool, TritonMaxPool1dK2S2NoIndices):
        return {"encoder_triton_pool_no_indices": True, "converted_triton_maxpool1d": 1}
    model.encoder.pool = TritonMaxPool1dK2S2NoIndices(pool)
    return {"encoder_triton_pool_no_indices": True, "converted_triton_maxpool1d": 1}


def replace_dna_embedder_block_with_fused_triton(model: nn.Module) -> dict[str, object]:
    """Replace encoder.dna_embedder.block with fused norm/gelu/int8-conv."""
    if triton is None:
        raise RuntimeError("triton is required for fused DNA embedder block replacement.")
    block = model.encoder.dna_embedder.block
    if isinstance(block, FusedNormGeluInt8WeightOnlyConv1d):
        return {"encoder_fused_dna_embedder_block": True, "converted_fused_convblocks": 1}
    model.encoder.dna_embedder.block = FusedNormGeluInt8WeightOnlyConv1d(block)
    return {"encoder_fused_dna_embedder_block": True, "converted_fused_convblocks": 1}


def replace_encoder_down_block0_with_fused_triton(model: nn.Module) -> dict[str, object]:
    """Replace encoder.down_blocks.0 with fused conv blocks and lean residual adds."""
    if triton is None:
        raise RuntimeError("triton is required for fused encoder down block replacement.")
    block = model.encoder.down_blocks[0]
    if isinstance(block, FusedDownResBlock0):
        return {"encoder_fused_down_block0": True, "converted_fused_down_blocks": 1}
    model.encoder.down_blocks[0] = FusedDownResBlock0(block)
    return {
        "encoder_fused_down_block0": True,
        "converted_fused_down_blocks": 1,
        "converted_fused_convblocks_in_down_blocks": 2,
    }


def convert_conv1d_to_triton_int8_weight_only(
    model: nn.Module,
    *,
    min_kernel_size: int = 2,
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = ("heads", "lora_", "locon_", "ia3", "adapter"),
    include_name_patterns: Iterable[str] = (),
) -> dict[str, object]:
    """Replace eligible Conv1d modules with the Triton int8 weight-only kernel."""
    if triton is None:
        raise RuntimeError("triton is required for custom int8 Conv1d quantization.")
    patterns = tuple(skip_name_patterns)
    include_patterns = tuple(include_name_patterns)
    decisions: dict[str, bool] = {}
    replacements: list[tuple[nn.Module, str, nn.Module]] = []

    for fqn, module, parent, child_name in _iter_named_child_modules(model):
        if not isinstance(module, nn.Conv1d):
            continue
        eligible = (
            module.__class__.__name__ != "StandardizedConv1d"
            and module.kernel_size[0] >= min_kernel_size
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
            replacements.append((parent, child_name, Int8WeightOnlyConv1d(module)))

    for parent, child_name, replacement in replacements:
        setattr(parent, child_name, replacement)

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return {
        "backend": "triton",
        "mode": "weight_only",
        "quant_type": "int8",
        "converted_triton_int8_conv1ds": converted,
        "skipped_triton_int8_conv1ds": skipped,
        "triton_int8_conv1d_skip_name_patterns": patterns,
        "triton_int8_conv1d_include_name_patterns": include_patterns,
    }


def convert_conv1d_to_triton_int8_dynamic_activation_int8_weight(
    model: nn.Module,
    *,
    min_kernel_size: int = 2,
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = ("heads", "lora_", "locon_", "ia3", "adapter"),
    include_name_patterns: Iterable[str] = (),
) -> dict[str, object]:
    """Replace eligible Conv1d modules with dynamic-int8 activation/weight kernels."""
    if triton is None:
        raise RuntimeError("triton is required for custom int8 Conv1d quantization.")
    patterns = tuple(skip_name_patterns)
    include_patterns = tuple(include_name_patterns)
    decisions: dict[str, bool] = {}
    replacements: list[tuple[nn.Module, str, nn.Module]] = []

    for fqn, module, parent, child_name in _iter_named_child_modules(model):
        if not isinstance(module, nn.Conv1d):
            continue
        eligible = (
            module.__class__.__name__ != "StandardizedConv1d"
            and module.kernel_size[0] >= min_kernel_size
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
            replacements.append((parent, child_name, Int8DynamicActivationInt8WeightConv1d(module)))

    for parent, child_name, replacement in replacements:
        setattr(parent, child_name, replacement)

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return {
        "backend": "triton",
        "mode": "dynamic_activation_weight",
        "quant_type": "int8",
        "converted_triton_int8_dynamic_conv1ds": converted,
        "skipped_triton_int8_dynamic_conv1ds": skipped,
        "triton_int8_dynamic_conv1d_skip_name_patterns": patterns,
        "triton_int8_dynamic_conv1d_include_name_patterns": include_patterns,
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
