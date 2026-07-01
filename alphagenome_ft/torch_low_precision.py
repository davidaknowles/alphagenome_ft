"""Local PyTorch low-precision conversion helpers for inference ablations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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


_CUDNN_INT8_CONV1D_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <sstream>

#define CHECK_CUDNN(expr) do { \
    cudnnStatus_t status = (expr); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::ostringstream oss; \
        oss << #expr << " failed: " << cudnnGetErrorString(status); \
        throw std::runtime_error(oss.str()); \
    } \
} while (0)

torch::Tensor cudnn_int8_conv1d_nhwc(torch::Tensor x, torch::Tensor w, int64_t pad_left) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kInt8 && w.scalar_type() == torch::kInt8, "x/w must be int8");
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4, "expected x NHWC and w OHWI tensors");
    x = x.contiguous();
    w = w.contiguous();
    int n = x.size(0);
    int h = x.size(1);
    int l = x.size(2);
    int c = x.size(3);
    int oc = w.size(0);
    int wh = w.size(1);
    int k = w.size(2);
    int wc = w.size(3);
    TORCH_CHECK(h == 1 && wh == 1 && wc == c, "cuDNN Conv1d NHWC/OHWI shape mismatch");

    auto y = torch::empty({n, 1, l, oc}, x.options().dtype(torch::kInt32));
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    CHECK_CUDNN(cudnnSetStream(handle, at::cuda::getCurrentCUDAStream().stream()));

    cudnnTensorDescriptor_t xd, yd;
    cudnnFilterDescriptor_t wd;
    cudnnConvolutionDescriptor_t cd;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xd));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yd));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wd));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&cd));

    CHECK_CUDNN(cudnnSetTensor4dDescriptorEx(xd, CUDNN_DATA_INT8, n, c, 1, l, l * c, 1, l * c, c));
    CHECK_CUDNN(cudnnSetTensor4dDescriptorEx(yd, CUDNN_DATA_INT32, n, oc, 1, l, l * oc, 1, l * oc, oc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_INT8, CUDNN_TENSOR_NHWC, oc, c, 1, k));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        cd, 0, static_cast<int>(pad_left), 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32));
    CHECK_CUDNN(cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH));

    int alpha = 1;
    int beta = 0;
    CHECK_CUDNN(cudnnConvolutionForward(
        handle,
        &alpha,
        xd,
        x.data_ptr(),
        wd,
        w.data_ptr(),
        cd,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        nullptr,
        0,
        &beta,
        yd,
        y.data_ptr()));

    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(cd));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wd));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yd));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xd));
    CHECK_CUDNN(cudnnDestroy(handle));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cudnn_int8_conv1d_nhwc", &cudnn_int8_conv1d_nhwc);
}
"""


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


class Conv1dAsConv2d(nn.Module):
    """Run a Conv1d as a Conv2d over a singleton height dimension."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        if conv.stride != (1,) or conv.dilation != (1,) or conv.groups != 1:
            raise ValueError("Conv1dAsConv2d only supports stride=1, dilation=1, groups=1.")
        self.pad_mode = getattr(conv, "pad_mode", conv.padding)
        self.kernel_size = conv.kernel_size
        self.conv2d = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=(1, conv.kernel_size[0]),
            stride=(1, conv.stride[0]),
            padding=(0, 0),
            dilation=(1, conv.dilation[0]),
            groups=conv.groups,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            self.conv2d.weight.copy_(conv.weight.unsqueeze(2))
            if conv.bias is not None:
                self.conv2d.bias.copy_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_mode == "same":
            pad_total = self.kernel_size[0] - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right))
        elif self.pad_mode not in {(0,), 0, "valid"}:
            raise ValueError(f"Unsupported Conv1dAsConv2d padding mode: {self.pad_mode!r}")
        return self.conv2d(x.unsqueeze(2)).squeeze(2)


def _nvidia_site_package_root() -> Path:
    return Path(torch.__file__).resolve().parents[1] / "nvidia"


@lru_cache(maxsize=1)
def _load_cudnn_int8_conv1d_extension():
    try:
        from torch.utils.cpp_extension import load_inline
    except ImportError as exc:  # pragma: no cover - optional build path
        raise RuntimeError("torch C++ extension support is required for cuDNN Conv1d.") from exc

    nvidia_root = _nvidia_site_package_root()
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    include_dirs = [
        nvidia_root / "cudnn" / "include",
        nvidia_root / "cuda_nvcc" / "include",
        nvidia_root / "cuda_cccl" / "include",
        nvidia_root / "cu13" / "include",
        nvidia_root / "cuda_runtime" / "include",
    ]
    library_dirs = [
        nvidia_root / "cudnn" / "lib",
        nvidia_root / "cu13" / "lib",
        nvidia_root / "cuda_runtime" / "lib",
        torch_lib,
    ]
    missing = [path for path in include_dirs + library_dirs if not path.exists()]
    cudnn_lib = nvidia_root / "cudnn" / "lib" / "libcudnn.so.9"
    if not cudnn_lib.exists():
        missing.append(cudnn_lib)
    if missing:
        raise RuntimeError(
            "Could not find the NVIDIA wheel headers/libs needed for cuDNN Conv1d: "
            + ", ".join(str(path) for path in missing)
        )

    return load_inline(
        name="ag_cudnn_int8_conv1d",
        cpp_sources=[_CUDNN_INT8_CONV1D_SOURCE],
        extra_include_paths=[str(path) for path in include_dirs],
        extra_ldflags=[
            *(f"-L{path}" for path in library_dirs),
            str(cudnn_lib),
            "-lc10_cuda",
            "-ltorch_cuda",
            *(f"-Wl,-rpath,{path}" for path in library_dirs),
        ],
        verbose=False,
    )


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


class CudnnInt8DynamicActivationInt8WeightConv1d(nn.Module):
    """cuDNN int8 activation/int8 weight Conv1d for NCL inference tensors."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        if conv.stride != (1,) or conv.dilation != (1,) or conv.groups != 1:
            raise ValueError(
                "CudnnInt8DynamicActivationInt8WeightConv1d only supports "
                "stride=1, dilation=1, groups=1."
            )
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.pad_mode = getattr(conv, "pad_mode", conv.padding)
        if self.pad_mode == "same":
            self.pad_left = (self.kernel_size[0] - 1) // 2
        elif self.pad_mode in {(0,), 0, "valid"}:
            self.pad_left = 0
        else:
            raise ValueError(f"Unsupported cuDNN Conv1d padding mode: {self.pad_mode!r}")

        weight = conv.weight.detach().float()
        flat = weight.reshape(weight.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        qweight = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("qweight_ohwi", qweight.reshape(*weight.shape).permute(0, 2, 1).unsqueeze(1).contiguous())
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
        batch, in_channels, length = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}.")

        x_float = x.float()
        x_absmax = x_float.abs().amax().clamp_min(1.0e-8)
        x_scale = x_absmax / 127.0
        x_quant = torch.round(x_float / x_scale).clamp(-127, 127).to(torch.int8)
        x_nhwc = x_quant.transpose(1, 2).unsqueeze(1).contiguous()

        ext = _load_cudnn_int8_conv1d_extension()
        y_int = ext.cudnn_int8_conv1d_nhwc(x_nhwc, self.qweight_ohwi, self.pad_left)
        y = y_int.float() * (x_scale * self.scale).view(1, 1, 1, -1)
        if self.has_bias:
            y = y + self.bias.view(1, 1, 1, -1)
        return y.squeeze(1).transpose(1, 2).contiguous().to(dtype=x.dtype)


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


def convert_conv1d_to_conv2d(
    model: nn.Module,
    *,
    min_kernel_size: int = 2,
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = ("heads", "lora_", "locon_", "ia3", "adapter"),
    include_name_patterns: Iterable[str] = (),
) -> dict[str, object]:
    """Replace eligible Conv1d modules with Conv2d wrappers.

    StandardizedConv1d modules are intentionally skipped. Materialize them into
    EffectiveConv1d first if their already-standardized inference weights should
    be eligible for Conv2d quantization.
    """
    if min_kernel_size < 1:
        raise ValueError("min_kernel_size must be >= 1")
    if min_feature_multiple < 1:
        raise ValueError("min_feature_multiple must be >= 1")

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
            replacements.append((parent, child_name, Conv1dAsConv2d(module)))

    for parent, child_name, replacement in replacements:
        setattr(parent, child_name, replacement)

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return {
        "converted_conv1d_to_conv2d": converted,
        "skipped_conv1d_to_conv2d": skipped,
        "conv2d_skip_name_patterns": patterns,
        "conv2d_include_name_patterns": include_patterns,
    }


def quantize_conv2ds_to_intx_weight_only(
    model: nn.Module,
    *,
    weight_dtype: torch.dtype,
    include_name_patterns: Iterable[str] = (),
) -> dict[str, object]:
    """Apply TorchAO intx weight-only quantization to eligible Conv2d modules."""
    try:
        from torchao.quantization import IntxWeightOnlyConfig, quantize_
    except ImportError as exc:
        raise RuntimeError("torchao is required for intx Conv2d quantization.") from exc

    include_patterns = tuple(include_name_patterns)
    decisions: dict[str, bool] = {}

    def module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        eligible = isinstance(mod, nn.Conv2d) and _matches_include_patterns(fqn, include_patterns)
        if isinstance(mod, nn.Conv2d):
            decisions[fqn] = eligible
        return eligible

    quantize_(model, IntxWeightOnlyConfig(weight_dtype=weight_dtype), filter_fn=module_filter_fn)
    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return {
        "backend": "torchao",
        "mode": "weight_only",
        "quant_type": str(weight_dtype).removeprefix("torch."),
        "converted_conv2ds": converted,
        "skipped_conv2ds": skipped,
        "conv2d_quant_include_name_patterns": include_patterns,
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


def convert_conv1d_to_cudnn_int8_dynamic_activation_int8_weight(
    model: nn.Module,
    *,
    min_kernel_size: int = 2,
    min_feature_multiple: int = 16,
    skip_name_patterns: Iterable[str] = ("heads", "lora_", "locon_", "ia3", "adapter"),
    include_name_patterns: Iterable[str] = (),
) -> dict[str, object]:
    """Replace eligible Conv1d modules with the experimental cuDNN int8 path."""
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
            replacements.append((parent, child_name, CudnnInt8DynamicActivationInt8WeightConv1d(module)))

    for parent, child_name, replacement in replacements:
        setattr(parent, child_name, replacement)

    converted = sum(1 for selected in decisions.values() if selected)
    skipped = sum(1 for selected in decisions.values() if not selected)
    return {
        "backend": "cudnn",
        "mode": "dynamic_activation_weight",
        "quant_type": "int8",
        "activation_scale": "tensorwise_dynamic",
        "weight_scale": "per_output_channel",
        "converted_cudnn_int8_dynamic_conv1ds": converted,
        "skipped_cudnn_int8_dynamic_conv1ds": skipped,
        "cudnn_int8_dynamic_conv1d_skip_name_patterns": patterns,
        "cudnn_int8_dynamic_conv1d_include_name_patterns": include_patterns,
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
