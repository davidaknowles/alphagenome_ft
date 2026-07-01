"""Local attention backend patches for AlphaGenome PyTorch inference."""

from __future__ import annotations

from functools import lru_cache
import math
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA path
    triton = None
    tl = None


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    try:
        from torch.nn.attention.flex_attention import flex_attention
    except ImportError as exc:  # pragma: no cover - optional PyTorch path
        raise RuntimeError("PyTorch FlexAttention is required for the flex attention backend.") from exc

    return torch.compile(flex_attention, dynamic=False)


def _mha_forward_flex(self, x, attention_bias, compute_dtype=None):
    from alphagenome_pytorch.attention import apply_rope

    batch, seq_len, _ = x.shape
    if compute_dtype is None:
        compute_dtype = x.dtype

    x = x.to(compute_dtype)
    h = self.norm(x)

    q = self.norm_q(self.q_proj(h).view(batch, seq_len, 8, 128))
    k = self.norm_k(self.k_proj(h).view(batch, seq_len, 1, 128))
    v = self.norm_v(self.v_proj(h).view(batch, seq_len, 1, 192))

    q = apply_rope(q, inplace=True)
    k = apply_rope(k, inplace=True)

    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    bias = attention_bias.float().contiguous()
    bias_is_lowres = bias.shape[-1] != seq_len
    logits_soft_cap = 5.0

    def score_mod(score, b, h_idx, q_idx, kv_idx):
        if bias_is_lowres:
            score = score + bias[b, h_idx, q_idx // 16, kv_idx // 16]
        else:
            score = score + bias[b, h_idx, q_idx, kv_idx]
        return torch.tanh(score / logits_soft_cap) * logits_soft_cap

    y = _compiled_flex_attention()(
        q_t,
        k_t,
        v_t,
        score_mod=score_mod,
        scale=1.0 / math.sqrt(128.0),
        enable_gqa=True,
        kernel_options={
            "BLOCK_M": 16,
            "BLOCK_N": 32,
            "num_warps": 4,
            "num_stages": 3,
        },
    )
    y = y.to(compute_dtype).permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
    y = self.linear_embedding(y)
    return self.final_norm(y)


if triton is not None:

    @triton.jit
    def _softcap_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        out_ptr,
        batch: tl.constexpr,
        heads: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        value_dim: tl.constexpr,
        bias_seq_len: tl.constexpr,
        bias_is_lowres: tl.constexpr,
        scale: tl.constexpr,
        soft_cap: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_dv = tl.program_id(2)

        b_idx = pid_bh // heads
        h_idx = pid_bh - b_idx * heads
        q_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_idx = tl.arange(0, BLOCK_D)
        dv_idx = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)

        q_offsets = ((b_idx * heads + h_idx) * seq_len + q_idx[:, None]) * head_dim + d_idx[None, :]
        q_mask = (q_idx[:, None] < seq_len) & (d_idx[None, :] < head_dim)
        q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_DV), tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            k_idx = start_n + tl.arange(0, BLOCK_N)
            k_offsets = (b_idx * seq_len + k_idx[None, :]) * head_dim + d_idx[:, None]
            k_mask = (k_idx[None, :] < seq_len) & (d_idx[:, None] < head_dim)
            k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
            scores = tl.dot(q, k, input_precision="tf32") * scale

            if bias_is_lowres:
                bias_q_idx = q_idx // 16
                bias_k_idx = k_idx // 16
            else:
                bias_q_idx = q_idx
                bias_k_idx = k_idx
            bias_offsets = ((b_idx * heads + h_idx) * bias_seq_len + bias_q_idx[:, None]) * bias_seq_len + bias_k_idx[None, :]
            score_mask = (q_idx[:, None] < seq_len) & (k_idx[None, :] < seq_len)
            bias = tl.load(bias_ptr + bias_offsets, mask=score_mask, other=-float("inf"))
            scores = scores + bias
            scores = (2.0 / (1.0 + tl.exp(-2.0 * scores / soft_cap)) - 1.0) * soft_cap
            scores = tl.where(score_mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            v_offsets = (b_idx * seq_len + k_idx[:, None]) * value_dim + dv_idx[None, :]
            v_mask = (k_idx[:, None] < seq_len) & (dv_idx[None, :] < value_dim)
            v = tl.load(v_ptr + v_offsets, mask=v_mask, other=0.0)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="tf32")
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        acc = acc / l_i[:, None]
        out_offsets = ((b_idx * heads + h_idx) * seq_len + q_idx[:, None]) * value_dim + dv_idx[None, :]
        out_mask = (q_idx[:, None] < seq_len) & (dv_idx[None, :] < value_dim)
        tl.store(out_ptr + out_offsets, acc, mask=out_mask)


    @triton.jit
    def _softcap_attention_fused_bias_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        pair_ptr,
        norm_inv_ptr,
        norm_bias_ptr,
        proj_weight_ptr,
        out_ptr,
        batch: tl.constexpr,
        heads: tl.constexpr,
        seq_len: tl.constexpr,
        pair_seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        value_dim: tl.constexpr,
        pair_dim: tl.constexpr,
        scale: tl.constexpr,
        soft_cap: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_dv = tl.program_id(2)

        b_idx = pid_bh // heads
        h_idx = pid_bh - b_idx * heads
        q_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_idx = tl.arange(0, BLOCK_D)
        dv_idx = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)

        q_offsets = ((b_idx * heads + h_idx) * seq_len + q_idx[:, None]) * head_dim + d_idx[None, :]
        q_mask = (q_idx[:, None] < seq_len) & (d_idx[None, :] < head_dim)
        q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_DV), tl.float32)
        sqrt_half = 0.7071067811865476

        for start_n in range(0, seq_len, BLOCK_N):
            k_idx = start_n + tl.arange(0, BLOCK_N)
            k_offsets = (b_idx * seq_len + k_idx[None, :]) * head_dim + d_idx[:, None]
            k_mask = (k_idx[None, :] < seq_len) & (d_idx[:, None] < head_dim)
            k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
            scores = tl.dot(q, k, input_precision="tf32") * scale

            pair_q_idx = q_idx // 16
            pair_k_idx = k_idx // 16
            bias = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for pair_d in range(0, pair_dim):
                pair_offsets = (
                    ((b_idx * pair_seq_len + pair_q_idx[:, None]) * pair_seq_len + pair_k_idx[None, :])
                    * pair_dim
                    + pair_d
                )
                pair_mask = (pair_q_idx[:, None] < pair_seq_len) & (pair_k_idx[None, :] < pair_seq_len)
                pair_val = tl.load(pair_ptr + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
                inv = tl.load(norm_inv_ptr + pair_d).to(tl.float32)
                nbias = tl.load(norm_bias_ptr + pair_d).to(tl.float32)
                x_norm = pair_val * inv + nbias
                gelu = 0.5 * x_norm * (1.0 + tl.erf(x_norm * sqrt_half))
                weight = tl.load(proj_weight_ptr + h_idx * pair_dim + pair_d).to(tl.float32)
                bias += gelu * weight

            score_mask = (q_idx[:, None] < seq_len) & (k_idx[None, :] < seq_len)
            scores = scores + bias
            scores = (2.0 / (1.0 + tl.exp(-2.0 * scores / soft_cap)) - 1.0) * soft_cap
            scores = tl.where(score_mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            v_offsets = (b_idx * seq_len + k_idx[:, None]) * value_dim + dv_idx[None, :]
            v_mask = (k_idx[:, None] < seq_len) & (dv_idx[None, :] < value_dim)
            v = tl.load(v_ptr + v_offsets, mask=v_mask, other=0.0)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="tf32")
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        acc = acc / l_i[:, None]
        out_offsets = ((b_idx * heads + h_idx) * seq_len + q_idx[:, None]) * value_dim + dv_idx[None, :]
        out_mask = (q_idx[:, None] < seq_len) & (dv_idx[None, :] < value_dim)
        tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def _triton_softcap_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is required for the custom attention backend.")
    if not (q.is_cuda and k.is_cuda and v.is_cuda and bias.is_cuda):
        raise RuntimeError("custom Triton attention requires CUDA tensors.")

    batch, heads, seq_len, head_dim = q.shape
    if k.shape != (batch, 1, seq_len, head_dim):
        raise ValueError(f"Expected K shape {(batch, 1, seq_len, head_dim)}, got {tuple(k.shape)}")
    value_dim = v.shape[-1]
    bias_seq_len = bias.shape[-1]
    bias_is_lowres = bias_seq_len != seq_len
    out = torch.empty((batch, heads, seq_len, value_dim), device=q.device, dtype=q.dtype)
    grid = (triton.cdiv(seq_len, 16), batch * heads, triton.cdiv(value_dim, 64))
    _softcap_attention_kernel[grid](
        q,
        k,
        v,
        bias,
        out,
        batch,
        heads,
        seq_len,
        head_dim,
        value_dim,
        bias_seq_len,
        bias_is_lowres,
        1.0 / math.sqrt(float(head_dim)),
        5.0,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_DV=64,
        BLOCK_D=128,
    )
    return out


def _triton_softcap_attention_fused_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_x: torch.Tensor,
    attention_bias_block: torch.nn.Module,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is required for the custom attention backend.")
    if not (q.is_cuda and k.is_cuda and v.is_cuda and pair_x.is_cuda):
        raise RuntimeError("custom Triton attention requires CUDA tensors.")

    batch, heads, seq_len, head_dim = q.shape
    pair_seq_len = pair_x.shape[1]
    value_dim = v.shape[-1]
    norm = attention_bias_block.norm
    norm_inv = (norm.weight * torch.rsqrt(norm.running_var + norm.eps).to(norm.weight.dtype)).contiguous()
    norm_bias = norm.bias.contiguous()
    proj_weight = attention_bias_block.proj.weight.contiguous()
    out = torch.empty((batch, heads, seq_len, value_dim), device=q.device, dtype=q.dtype)
    grid = (triton.cdiv(seq_len, 16), batch * heads, triton.cdiv(value_dim, 64))
    _softcap_attention_fused_bias_kernel[grid](
        q,
        k,
        v,
        pair_x.contiguous(),
        norm_inv,
        norm_bias,
        proj_weight,
        out,
        batch,
        heads,
        seq_len,
        pair_seq_len,
        head_dim,
        value_dim,
        pair_x.shape[-1],
        1.0 / math.sqrt(float(head_dim)),
        5.0,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_DV=64,
        BLOCK_D=128,
    )
    return out


def _mha_forward_triton(self, x, attention_bias, compute_dtype=None):
    from alphagenome_pytorch.attention import apply_rope

    batch, seq_len, _ = x.shape
    if compute_dtype is None:
        compute_dtype = x.dtype

    x = x.to(compute_dtype)
    h = self.norm(x)

    q = self.norm_q(self.q_proj(h).view(batch, seq_len, 8, 128))
    k = self.norm_k(self.k_proj(h).view(batch, seq_len, 1, 128))
    v = self.norm_v(self.v_proj(h).view(batch, seq_len, 1, 192))

    q = apply_rope(q, inplace=True)
    k = apply_rope(k, inplace=True)

    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    y = _triton_softcap_attention(q_t, k_t, v_t, attention_bias.float().contiguous())
    y = y.to(compute_dtype).permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
    y = self.linear_embedding(y)
    return self.final_norm(y)


def _mha_forward_triton_fused_bias(self, x, attention_payload, compute_dtype=None):
    from alphagenome_pytorch.attention import apply_rope

    pair_x, attention_bias_block = attention_payload
    batch, seq_len, _ = x.shape
    if compute_dtype is None:
        compute_dtype = x.dtype

    x = x.to(compute_dtype)
    h = self.norm(x)

    q = self.norm_q(self.q_proj(h).view(batch, seq_len, 8, 128))
    k = self.norm_k(self.k_proj(h).view(batch, seq_len, 1, 128))
    v = self.norm_v(self.v_proj(h).view(batch, seq_len, 1, 192))

    q = apply_rope(q, inplace=True)
    k = apply_rope(k, inplace=True)

    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    y = _triton_softcap_attention_fused_bias(
        q_t,
        k_t,
        v_t,
        pair_x.to(compute_dtype),
        attention_bias_block,
    )
    y = y.to(compute_dtype).permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
    y = self.linear_embedding(y)
    return self.final_norm(y)


def _attention_bias_forward_lowres(self, x):
    h = F.gelu(self.norm(x))
    h = self.proj(h)
    return h.permute(0, 3, 1, 2).contiguous()


def _tower_forward_block_fused_bias(self, block, x, pair_x, compute_dtype):
    if block["pair_update"] is not None:
        pair_x = block["pair_update"](x, pair_x, compute_dtype=compute_dtype)
    x = x + block["mha"](x, (pair_x, block["attn_bias"]), compute_dtype=compute_dtype)
    x = x + block["mlp"](x)
    return x, pair_x


def apply_attention_backend(model: torch.nn.Module, backend: str) -> dict[str, Any]:
    """Patch AlphaGenome MHA modules in-place for inference benchmarks."""
    if backend not in {
        "flex_mha",
        "triton_mha",
        "flex_mha_lowres_bias",
        "triton_mha_lowres_bias",
        "triton_mha_fused_bias",
    }:
        raise ValueError(f"Unsupported attention backend: {backend!r}")

    patched = 0
    patched_bias = 0
    for module in model.modules():
        if module.__class__.__name__ == "AttentionBiasBlock" and backend.endswith("_lowres_bias"):
            module.forward = MethodType(_attention_bias_forward_lowres, module)
            patched_bias += 1
            continue
        if module.__class__.__name__ != "MHABlock":
            continue
        if backend in {"flex_mha", "flex_mha_lowres_bias"}:
            module.forward = MethodType(_mha_forward_flex, module)
        elif backend == "triton_mha_fused_bias":
            module.forward = MethodType(_mha_forward_triton_fused_bias, module)
        else:
            module.forward = MethodType(_mha_forward_triton, module)
        patched += 1

    patched_tower = 0
    if backend == "triton_mha_fused_bias":
        for module in model.modules():
            if module.__class__.__name__ == "TransformerTower":
                module._forward_block = MethodType(_tower_forward_block_fused_bias, module)
                patched_tower += 1

    return {
        "attention_backend": backend,
        "patched_mha_blocks": patched,
        "patched_attention_bias_blocks": patched_bias,
        "patched_transformer_towers": patched_tower,
        "row_attention_backend": "eager",
        "attention_bias_resolution": "fused"
        if backend.endswith("_fused_bias")
        else "lowres"
        if backend.endswith("_lowres_bias")
        else "full",
    }
