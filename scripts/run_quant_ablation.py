#!/usr/bin/env python
"""Evaluate merged AlphaGenome checkpoints under implemented quantized inference paths."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


DEFAULT_BIGWIG_DIR = Path(
    "/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs"
)
DEFAULT_FASTA = Path("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa")
DEFAULT_MERGED = Path("outputs/quant_ablation/merged_jax_default_lora_locon")

TORCH_TRUE_QUANT_STRATEGIES: dict[str, dict[str, Any]] = {
    "torchao_float8_linear": {"kind": "float8", "include": (), "skip": None},
    "torchao_float8_tower_linear": {"kind": "float8", "include": ("tower",), "skip": None},
    "torchao_float8_all_linear": {"kind": "float8", "include": (), "skip": ()},
    "torchao_float8_linear_1x1conv": {
        "kind": "float8",
        "include": (),
        "skip": None,
        "pointwise_conv": True,
    },
    "torchao_float8_all_linear_1x1conv": {
        "kind": "float8",
        "include": (),
        "skip": (),
        "pointwise_conv": True,
    },
    "torchao_nvfp4_weight_only_linear": {"kind": "nvfp4", "include": (), "skip": None},
    "torchao_nvfp4_weight_only_tower_linear": {
        "kind": "nvfp4",
        "include": ("tower",),
        "skip": None,
    },
    "torchao_nvfp4_weight_only_all_linear": {"kind": "nvfp4", "include": (), "skip": ()},
    "torchao_nvfp4_weight_only_linear_1x1conv": {
        "kind": "nvfp4",
        "include": (),
        "skip": None,
        "pointwise_conv": True,
    },
    "torchao_nvfp4_weight_only_all_linear_1x1conv": {
        "kind": "nvfp4",
        "include": (),
        "skip": (),
        "pointwise_conv": True,
    },
    "bnb_nf4_weight_only_linear": {"kind": "bnb_nf4", "include": (), "skip": None},
    "bnb_nf4_weight_only_tower_linear": {"kind": "bnb_nf4", "include": ("tower",), "skip": None},
    "bnb_nf4_weight_only_all_linear": {"kind": "bnb_nf4", "include": (), "skip": ()},
    "bnb_nf4_weight_only_linear_1x1conv": {
        "kind": "bnb_nf4",
        "include": (),
        "skip": None,
        "pointwise_conv": True,
    },
    "bnb_nf4_weight_only_all_linear_1x1conv": {
        "kind": "bnb_nf4",
        "include": (),
        "skip": (),
        "pointwise_conv": True,
    },
    "triton_int8_weight_only_conv1d": {
        "kind": "triton_int8_conv1d",
        "include": (),
    },
    "triton_int8_weight_only_effective_conv1d": {
        "kind": "triton_int8_conv1d",
        "include": ("encoder.down_blocks.4", "encoder.down_blocks.5"),
    },
    "triton_int8_dynamic_conv1d": {
        "kind": "triton_int8_dynamic_conv1d",
        "include": (),
    },
    "triton_int8_dynamic_effective_conv1d": {
        "kind": "triton_int8_dynamic_conv1d",
        "include": ("encoder.down_blocks.4", "encoder.down_blocks.5"),
    },
    "bf16_triton_conv": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
    },
    "bf16_triton_conv_no_intermediates": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "encoder_no_intermediates": True,
    },
    "bf16_triton_conv_no_intermediates_tritonpool": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "encoder_no_intermediates": True,
        "encoder_triton_pool": True,
    },
    "bf16_triton_conv_no_intermediates_tritonpool_fusedembed": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "encoder_no_intermediates": True,
        "encoder_triton_pool": True,
        "encoder_fused_dna_embedder_block": True,
    },
    "bf16_triton_conv_no_intermediates_tritonpool_fuseddown0": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "encoder_no_intermediates": True,
        "encoder_triton_pool": True,
        "encoder_fused_down_block0": True,
    },
    "bf16_triton_conv_no_intermediates_tritonpool_fusedembed_fuseddown0": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "encoder_no_intermediates": True,
        "encoder_triton_pool": True,
        "encoder_fused_dna_embedder_block": True,
        "encoder_fused_down_block0": True,
    },
    "bf16_triton_conv_no_intermediates_tritonpool_fusedembed_fuseddown0_flexattn_lowresbias": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "encoder_no_intermediates": True,
        "encoder_triton_pool": True,
        "encoder_fused_dna_embedder_block": True,
        "encoder_fused_down_block0": True,
        "attention_backend": "flex_mha_lowres_bias",
    },
    "bf16_triton_conv_flexattn": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "attention_backend": "flex_mha",
    },
    "bf16_triton_conv_tritonattn": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "attention_backend": "triton_mha",
    },
    "bf16_triton_conv_flexattn_lowresbias": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "attention_backend": "flex_mha_lowres_bias",
    },
    "bf16_triton_conv_tritonattn_lowresbias": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "attention_backend": "triton_mha_lowres_bias",
    },
    "bf16_triton_conv_tritonattn_fusedbias": {
        "kind": "triton_int8_conv1d",
        "include": (),
        "bf16_params": True,
        "attention_backend": "triton_mha_fused_bias",
    },
    "nvfp4_linear1x1_triton_conv": {
        "kind": "nvfp4_triton_int8_conv1d",
        "include": (),
        "skip": (),
        "pointwise_conv": True,
        "conv_include": (),
        "bf16_params": True,
    },
    "nf4_linear1x1_triton_conv": {
        "kind": "bnb_nf4_triton_int8_conv1d",
        "include": (),
        "skip": (),
        "pointwise_conv": True,
        "conv_include": (),
        "bf16_params": True,
    },
    "torchao_nvfp4_weight_only_all_linear_1x1conv_triton_int8_weight_only_conv1d": {
        "kind": "nvfp4_triton_int8_conv1d",
        "include": (),
        "skip": (),
        "pointwise_conv": True,
        "conv_include": (),
        "bf16_params": True,
    },
    "bnb_nf4_weight_only_all_linear_1x1conv_triton_int8_weight_only_conv1d": {
        "kind": "bnb_nf4_triton_int8_conv1d",
        "include": (),
        "skip": (),
        "pointwise_conv": True,
        "conv_include": (),
        "bf16_params": True,
    },
}

TORCH_STDCONV_EFFECTIVE_SUFFIX = "_stdconv_effective"


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _keypath_to_str(path: tuple) -> str:
    return "/".join(str(getattr(key, "key", key)) for key in path)


def _is_float_array(value: Any) -> bool:
    import jax.numpy as jnp

    return hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)


def _load_local_torch_low_precision():
    module_path = REPO_ROOT / "alphagenome_ft" / "torch_low_precision.py"
    spec = importlib.util.spec_from_file_location("_agft_torch_low_precision", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local low precision helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_agft_torch_low_precision", module)
    spec.loader.exec_module(module)
    return module


def _load_local_torch_attention_backends():
    module_path = REPO_ROOT / "alphagenome_ft" / "torch_attention_backends.py"
    spec = importlib.util.spec_from_file_location("_agft_torch_attention_backends", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local attention helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_agft_torch_attention_backends", module)
    spec.loader.exec_module(module)
    return module


def _load_torch_checkpoint_payload(weights_path, *, map_location="cpu"):
    import torch

    if weights_path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        metadata = {}
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            raw_metadata = handle.metadata() or {}
        if "alphagenome_fp4_metadata" in raw_metadata:
            metadata = json.loads(raw_metadata["alphagenome_fp4_metadata"])
        else:
            sidecar = weights_path.with_suffix(weights_path.suffix + ".json")
            if sidecar.exists():
                with sidecar.open() as handle:
                    metadata = json.load(handle)
        return {
            **metadata,
            "model_state_dict": load_file(str(weights_path), device=str(map_location)),
        }

    return torch.load(weights_path, map_location=map_location, weights_only=False)


def _drop_encoder_intermediates_for_128bp_eval(model: Any) -> dict[str, Any]:
    """Patch the Torch encoder to skip U-Net skip tensors for 128bp-only eval."""
    import types

    encoder = model.encoder
    if getattr(encoder, "_alphagenome_fp4_no_intermediates", False):
        return {"encoder_no_intermediates": True}

    def forward_no_intermediates(self: Any, x: Any):
        x = x.transpose(1, 2)
        x = self.dna_embedder(x)
        x = self.pool(x)
        for block in self.down_blocks:
            x = block(x)
            x = self.pool(x)
        return x, {}

    encoder.forward = types.MethodType(forward_no_intermediates, encoder)
    encoder._alphagenome_fp4_no_intermediates = True
    return {"encoder_no_intermediates": True}


def _split_torch_strategy(strategy: str) -> tuple[str, bool]:
    if strategy.endswith(TORCH_STDCONV_EFFECTIVE_SUFFIX):
        return strategy[: -len(TORCH_STDCONV_EFFECTIVE_SUFFIX)], True
    return strategy, False


def _path_is_sensitive(path: str) -> bool:
    lowered = path.lower()
    sensitive_tokens = (
        "/head/",
        "head/",
        "norm",
        "batch_norm",
        "rms",
        "embedding",
        "embed/",
        "scale",
        "bias",
        "/b",
        "organism_embed",
    )
    return any(token in lowered for token in sensitive_tokens)


def apply_jax_quant_policy(params, strategy: str, *, nf4_block_size: int = 64):
    import jax

    if strategy == "default":
        return params, {"strategy": strategy, "converted": 0, "simulated_storage": False}
    if strategy not in {"bf16_params"}:
        raise ValueError(
            f"Unsupported JAX strategy {strategy!r}. JAX NF4/FP8 ablations were "
            "roundtrip simulations; use the torch backend for implemented true quantized paths."
        )
    converted_paths: list[str] = []

    def convert(path_tuple, value):
        path = _keypath_to_str(path_tuple)
        if strategy == "bf16_params":
            if _is_float_array(value) and not _path_is_sensitive(path):
                converted_paths.append(path)
                return value.astype(jnp.bfloat16)
            return value
        return value

    converted = jax.tree_util.tree_map_with_path(convert, params)
    return converted, {
        "strategy": strategy,
        "converted": len(converted_paths),
        "converted_paths": converted_paths[:200],
        "converted_paths_truncated": len(converted_paths) > 200,
        "simulated_storage": strategy.startswith(("nf4_", "fp8_")),
    }


def apply_torch_quant_policy(model: Any, strategy: str, *, nf4_block_size: int = 64) -> dict[str, Any]:
    if strategy == "default":
        return {"strategy": strategy, "converted": 0, "simulated_storage": False}
    if strategy == "bf16_params":
        import torch

        model.to(dtype=torch.bfloat16)
        return {"strategy": strategy, "converted": -1, "simulated_storage": False}
    if strategy not in TORCH_TRUE_QUANT_STRATEGIES:
        raise ValueError(
            f"Unsupported torch strategy {strategy!r}. Supported true quant strategies: "
            f"{', '.join(sorted(TORCH_TRUE_QUANT_STRATEGIES))}."
        )

    config = TORCH_TRUE_QUANT_STRATEGIES[strategy]
    include_patterns = tuple(config["include"])
    skip_patterns = config.get("skip")
    kwargs = {"skip_name_patterns": tuple(skip_patterns)} if skip_patterns is not None else {}
    if config.get("bf16_params"):
        import torch

        model.to(dtype=torch.bfloat16)
    encoder_stats = {}
    if config.get("encoder_no_intermediates"):
        encoder_stats = _drop_encoder_intermediates_for_128bp_eval(model)
    pool_stats = {}
    if config.get("encoder_triton_pool"):
        low_precision = _load_local_torch_low_precision()
        pool_stats = low_precision.replace_encoder_pool_with_triton_no_indices(model)
    fused_block_stats = {}
    if config.get("encoder_fused_dna_embedder_block"):
        low_precision = _load_local_torch_low_precision()
        fused_block_stats = low_precision.replace_dna_embedder_block_with_fused_triton(model)
    fused_down_stats = {}
    if config.get("encoder_fused_down_block0"):
        low_precision = _load_local_torch_low_precision()
        fused_down_stats = low_precision.replace_encoder_down_block0_with_fused_triton(model)
    attention_stats = {}
    if config.get("attention_backend"):
        attention_backends = _load_local_torch_attention_backends()
        attention_stats = attention_backends.apply_attention_backend(
            model,
            str(config["attention_backend"]),
        )
    pointwise_conv_stats = {}
    if config.get("pointwise_conv"):
        low_precision = _load_local_torch_low_precision()
        pointwise_conv_stats = low_precision.convert_pointwise_conv1d_to_linear(
            model,
            include_name_patterns=include_patterns,
            **kwargs,
        )
    kind = str(config["kind"])
    if kind == "float8":
        low_precision = _load_local_torch_low_precision()

        stats = low_precision.convert_linears_to_float8_training(
            model, recipe="tensorwise", include_name_patterns=include_patterns, **kwargs
        )
        return {
            **stats.__dict__,
            **pointwise_conv_stats,
            "strategy": strategy,
            "converted": stats.converted_linears,
            "simulated_storage": False,
        }
    if kind == "nvfp4":
        low_precision = _load_local_torch_low_precision()

        for param in model.parameters():
            param.requires_grad_(False)
        stats = low_precision.convert_linears_to_nvfp4_weight_only(
            model,
            include_name_patterns=include_patterns,
            **kwargs,
        )
        return {
            **stats.__dict__,
            **pointwise_conv_stats,
            "strategy": strategy,
            "converted": stats.converted_linears,
            "simulated_storage": False,
        }
    if kind == "bnb_nf4":
        import torch

        low_precision = _load_local_torch_low_precision()

        for param in model.parameters():
            param.requires_grad_(False)
        stats = low_precision.convert_linears_to_bnb_nf4_weight_only(
            model,
            compute_dtype=torch.bfloat16,
            include_name_patterns=include_patterns,
            **kwargs,
        )
        return {
            **stats.__dict__,
            **pointwise_conv_stats,
            "strategy": strategy,
            "converted": stats.converted_linears,
            "simulated_storage": False,
        }
    if kind == "triton_int8_conv1d":
        low_precision = _load_local_torch_low_precision()
        stats = low_precision.convert_conv1d_to_triton_int8_weight_only(
            model,
            include_name_patterns=include_patterns,
        )
        return {
            **stats,
            **encoder_stats,
            **pool_stats,
            **fused_block_stats,
            **fused_down_stats,
            **attention_stats,
            "strategy": strategy,
            "converted": stats["converted_triton_int8_conv1ds"],
            "bf16_params": bool(config.get("bf16_params", False)),
            "simulated_storage": False,
        }
    if kind == "triton_int8_dynamic_conv1d":
        low_precision = _load_local_torch_low_precision()
        stats = low_precision.convert_conv1d_to_triton_int8_dynamic_activation_int8_weight(
            model,
            include_name_patterns=include_patterns,
        )
        return {
            **stats,
            "strategy": strategy,
            "converted": stats["converted_triton_int8_dynamic_conv1ds"],
            "simulated_storage": False,
        }
    if kind == "nvfp4_triton_int8_conv1d":
        low_precision = _load_local_torch_low_precision()

        for param in model.parameters():
            param.requires_grad_(False)
        linear_stats = low_precision.convert_linears_to_nvfp4_weight_only(
            model,
            include_name_patterns=include_patterns,
            **kwargs,
        )
        conv_stats = low_precision.convert_conv1d_to_triton_int8_weight_only(
            model,
            include_name_patterns=tuple(config["conv_include"]),
        )
        return {
            **linear_stats.__dict__,
            **pointwise_conv_stats,
            **conv_stats,
            "strategy": strategy,
            "converted": linear_stats.converted_linears
            + conv_stats["converted_triton_int8_conv1ds"],
            "bf16_params": bool(config.get("bf16_params", False)),
            "simulated_storage": False,
        }
    if kind == "bnb_nf4_triton_int8_conv1d":
        import torch

        low_precision = _load_local_torch_low_precision()

        for param in model.parameters():
            param.requires_grad_(False)
        linear_stats = low_precision.convert_linears_to_bnb_nf4_weight_only(
            model,
            compute_dtype=torch.bfloat16,
            include_name_patterns=include_patterns,
            **kwargs,
        )
        conv_stats = low_precision.convert_conv1d_to_triton_int8_weight_only(
            model,
            include_name_patterns=tuple(config["conv_include"]),
        )
        return {
            **linear_stats.__dict__,
            **pointwise_conv_stats,
            **conv_stats,
            "strategy": strategy,
            "converted": linear_stats.converted_linears
            + conv_stats["converted_triton_int8_conv1ds"],
            "bf16_params": bool(config.get("bf16_params", False)),
            "simulated_storage": False,
        }
    raise AssertionError(f"Unhandled strategy config: {strategy}")


class GpuSampler:
    def __init__(self, path: Path, interval_sec: float = 1.0):
        self.path = path
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["time", "util_pct", "mem_mib"])
            writer.writeheader()
            while not self._stop.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    first = result.stdout.strip().splitlines()[0]
                    util, mem = [item.strip() for item in first.split(",")[:2]]
                    writer.writerow(
                        {"time": time.time(), "util_pct": float(util), "mem_mib": float(mem)}
                    )
                    handle.flush()
                except Exception:
                    pass
                self._stop.wait(self.interval_sec)


def _gpu_summary(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {"samples": 0, "avg_util_pct": None, "nonzero_util_pct": None, "max_mem_mib": None}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"samples": 0, "avg_util_pct": None, "nonzero_util_pct": None, "max_mem_mib": None}
    utils = np.asarray([float(row["util_pct"]) for row in rows], dtype=np.float64)
    mem = np.asarray([float(row["mem_mib"]) for row in rows], dtype=np.float64)
    return {
        "samples": int(len(rows)),
        "avg_util_pct": float(np.mean(utils)),
        "nonzero_util_pct": float(np.mean(utils > 0.0) * 100.0),
        "max_mem_mib": float(np.max(mem)),
    }


def _maybe_bin_128bp_np(values: np.ndarray) -> np.ndarray:
    if values.ndim == 3 and values.shape[1] >= 8192 and values.shape[1] % 128 == 0:
        return values.reshape(values.shape[0], values.shape[1] // 128, 128, values.shape[2]).mean(axis=2)
    return values


def _double_center_pearson_np(prediction: np.ndarray, targets: np.ndarray) -> float:
    prediction = _maybe_bin_128bp_np(prediction)
    targets = _maybe_bin_128bp_np(targets)
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


def r2_metrics_np(prediction: Any, targets: Any) -> dict[str, float]:
    y = np.asarray(targets, dtype=np.float32)
    yhat = np.asarray(prediction, dtype=np.float32)
    residual = yhat - y

    sse = float(np.sum(np.square(residual)))
    centered = y - float(np.mean(y))
    sst = float(np.sum(np.square(centered)))
    r2_global = float(1.0 - sse / sst) if sst > 0 else float("nan")

    target_mean_by_locus = np.mean(y, axis=-1, keepdims=True)
    differential_pearson_r = _double_center_pearson_np(yhat, y)

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


_DNA_ONE_HOT_LOOKUP = np.zeros((256, 4), dtype=np.float32)
for _base, _col in (
    (b"A", 0),
    (b"C", 1),
    (b"G", 2),
    (b"T", 3),
    (b"a", 0),
    (b"c", 1),
    (b"g", 2),
    (b"t", 3),
):
    _DNA_ONE_HOT_LOOKUP[_base[0], _col] = 1.0


def _one_hot_dna(sequence: str) -> np.ndarray:
    ascii_sequence = np.frombuffer(sequence.encode("ascii", errors="ignore"), dtype=np.uint8)
    if ascii_sequence.shape[0] != len(sequence):
        raise ValueError("Expected ASCII DNA sequence from FASTA.")
    return _DNA_ONE_HOT_LOOKUP[ascii_sequence].copy()


class CachedTorchDataModule:
    """Minimal cached-target data module for Torch inference ablations."""

    def __init__(
        self,
        *,
        target_cache_dir: Path,
        fasta_path: Path,
        head_id: str,
        batch_size: int,
    ) -> None:
        if target_cache_dir is None:
            raise ValueError("CachedTorchDataModule requires --target-cache-dir.")
        try:
            from pyfaidx import Fasta
        except ImportError as exc:
            raise RuntimeError("pyfaidx is required for Torch cached ablations.") from exc

        self.cache_dir = target_cache_dir.expanduser().resolve()
        with (self.cache_dir / "manifest.json").open() as handle:
            self.manifest = json.load(handle)
        self.fasta = Fasta(str(fasta_path.expanduser().resolve()), as_raw=True, sequence_always_upper=True)
        self.head_id = head_id
        self.batch_size = batch_size
        tracks = self.manifest["heads"][head_id]["tracks"]
        self.head_specs = [type("TorchHeadSpec", (), {"head_id": head_id, "tracks": tracks})()]
        self._arrays: dict[str, np.memmap] = {}

    def iter_batches(self, split: str, *, shuffle: bool = False) -> Sequence[dict[str, np.ndarray]]:
        if shuffle:
            raise ValueError("Cached Torch ablation iterator does not support shuffle.")
        intervals = self.manifest["splits"].get(split, [])
        if not intervals:
            return
        array = self._arrays.get(split)
        if array is None:
            array = np.load(self.cache_dir / split / f"{self.head_id}.npy", mmap_mode="r")
            self._arrays[split] = array

        for start_idx in range(0, len(intervals), self.batch_size):
            batch_intervals = intervals[start_idx : start_idx + self.batch_size]
            sequences = []
            for interval in batch_intervals:
                chrom = interval["chromosome"]
                start = int(interval["start"])
                end = int(interval["end"])
                sequences.append(_one_hot_dna(self.fasta[chrom][start:end]))
            batch_indices = slice(start_idx, start_idx + len(batch_intervals))
            yield {
                "sequences": np.stack(sequences, axis=0),
                f"targets_{self.head_id}": np.asarray(array[batch_indices]),
                "negative_strand_mask": np.zeros((len(batch_intervals),), dtype=bool),
            }


def build_data_module(args: argparse.Namespace):
    from alphagenome_ft.finetune import (
        BigWigDataModule,
        WindowedTargetCache,
        load_targets_config,
        prepare_head_specs,
        register_predefined_heads,
        validate_head_specs,
    )
    from scripts.run_humanbraindev_finetune import (
        build_targets_config,
        discover_bigwigs,
        make_chromosome_split_intervals,
        parse_chrom_set,
    )

    bigwigs = discover_bigwigs(args.bigwig_dir.expanduser().resolve())
    targets_config = load_targets_config(build_targets_config(bigwigs, args.head_id))
    head_specs = prepare_head_specs(targets_config, organism="HOMO_SAPIENS")
    validate_head_specs(head_specs)
    register_predefined_heads(head_specs)
    intervals = make_chromosome_split_intervals(
        args.fasta_path.expanduser().resolve(),
        window_size=args.window_size,
        stride=args.stride,
        valid_chroms=parse_chrom_set(args.valid_chroms),
        test_chroms=parse_chrom_set(args.test_chroms),
        exclude_chroms=parse_chrom_set(args.exclude_chroms),
        limit_train=args.limit_train,
        limit_valid=args.limit_valid,
        limit_test=args.limit_test,
    )
    target_cache_dir = args.target_cache_dir.expanduser().resolve() if args.target_cache_dir else None
    if args.build_target_cache:
        WindowedTargetCache.build(
            target_cache_dir,
            intervals=BigWigDataModule._filter_intervals_by_bigwig_chromosomes(intervals, head_specs),
            head_specs=head_specs,
            dtype=args.target_cache_dtype,
            workers=args.target_cache_workers or max(1, os.cpu_count() or 1),
            overwrite=False,
        )
    module = BigWigDataModule(
        intervals=intervals,
        fasta_path=args.fasta_path.expanduser().resolve(),
        head_specs=head_specs,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        target_workers=args.target_workers,
        window_workers=args.window_workers,
        target_cache_dir=target_cache_dir,
        target_cache_dtype=args.target_cache_dtype,
    )
    return module, head_specs


def load_merged_jax_model(args: argparse.Namespace, head_names: Sequence[str]):
    """Load a full merged checkpoint into a custom-head model template."""
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    checkpoint_root = args.checkpoint.expanduser().resolve()
    checkpoint_path = checkpoint_root / "checkpoint"
    config_path = checkpoint_root / "config.json"
    checkpoint_config: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open() as handle:
            checkpoint_config = json.load(handle)
    effective_conv_paths = tuple(
        str(path)
        for path in (
            checkpoint_config.get("backbone_effective_conv_paths")
            or checkpoint_config.get("effective_conv_paths")
            or ()
        )
    )
    model = create_model_with_heads(
        args.model_version,
        heads=tuple(head_names),
        checkpoint_path=args.base_checkpoint_path.expanduser().resolve()
        if args.base_checkpoint_path
        else None,
        detach_backbone=True,
        init_seq_len=args.window_size,
        backbone_effective_conv_paths=effective_conv_paths,
    )
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(str(checkpoint_path), target=(model._params, model._state))
    if isinstance(restored, (tuple, list)) and len(restored) == 2:
        model._params, model._state = restored
    else:
        raise ValueError(f"Unexpected checkpoint payload from {checkpoint_path!s}")
    return model


def evaluate_jax(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import numpy as np
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import dna_model as research_dna_model
    from alphagenome_ft.finetune.data import prepare_batch
    from alphagenome_ft.finetune.train import _add_stats, _finalize_r2_stats, _r2_stats

    data_module, head_specs = build_data_module(args)
    head_names = tuple(spec.head_id for spec in head_specs)
    model = load_merged_jax_model(args, head_names)
    model._params, quant_stats = apply_jax_quant_policy(
        model._params,
        args.strategy,
        nf4_block_size=args.nf4_block_size,
    )
    organism_enum = getattr(ag_dna_model.Organism, "HOMO_SAPIENS")
    organism_index_value = research_dna_model.convert_to_organism_index(organism_enum)
    strand_reindexing = model._metadata[organism_enum].strand_reindexing
    loss_fns = {name: model.create_loss_fn_for_head(name) for name in head_names}

    @jax.jit
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
        losses = {}
        stats = {}
        for head_name in head_names:
            targets = batch[f"targets_{head_name}"]
            losses[head_name] = loss_fns[head_name](
                predictions[head_name],
                {"targets": targets, "organism_index": batch["organism_index"]},
            )["loss"]
            stats[head_name] = _r2_stats(predictions[head_name], targets)
        return losses, stats

    split_results: dict[str, Any] = {}
    total_batches = 0
    total_examples = 0
    start_time = time.perf_counter()
    with model._device_context:
        for split in args.splits.split(","):
            split = split.strip()
            if not split:
                continue
            losses = {head: [] for head in head_names}
            stats_by_head: dict[str, dict[str, Any] | None] = {head: None for head in head_names}
            batches = 0
            examples = 0
            for batch_np in data_module.iter_batches(split, shuffle=False):
                batch_examples = int(batch_np["sequences"].shape[0])
                batch = prepare_batch(batch_np, organism_index_value, head_names)
                batch["strand_reindexing"] = strand_reindexing
                head_losses, head_stats = eval_step(model._params, model._state, batch)
                for head_name in head_names:
                    losses[head_name].append(float(np.asarray(head_losses[head_name])))
                    stats_by_head[head_name] = _add_stats(stats_by_head[head_name], head_stats[head_name])
                batches += 1
                total_batches += 1
                examples += batch_examples
                total_examples += batch_examples
                if args.max_batches and batches >= args.max_batches:
                    break
            split_results[split] = {}
            for head_name in head_names:
                metrics = {
                    "loss": float(np.mean(losses[head_name])) if losses[head_name] else float("nan"),
                    "batches": batches,
                    "examples": examples,
                }
                if stats_by_head[head_name] is not None:
                    metrics.update(_finalize_r2_stats(stats_by_head[head_name]))
                split_results[split][head_name] = metrics
    elapsed = time.perf_counter() - start_time
    return {
        "backend": "jax",
        "strategy": args.strategy,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "splits": split_results,
        "quantization": quant_stats,
        "elapsed_sec": elapsed,
        "batch_size": args.batch_size,
        "batches": total_batches,
        "batches_per_sec": total_batches / elapsed if elapsed > 0 else None,
        "examples": total_examples,
        "examples_per_sec": total_examples / elapsed if elapsed > 0 else None,
    }


def evaluate_torch(args: argparse.Namespace) -> dict[str, Any]:
    torch_repo = args.torch_repo.expanduser().resolve()
    src_dir = torch_repo / "src"
    for path in (str(src_dir), str(torch_repo)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch
    import torch.nn.functional as F
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.model import AlphaGenome
    from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
    from alphagenome_pytorch.extensions.finetuning.transfer import add_head, remove_all_heads
    from torch_effective_conv import (
        jax_effective_paths_to_torch,
        materialize_effective_convs,
        materialize_standardized_convs,
    )

    data_module = CachedTorchDataModule(
        target_cache_dir=args.target_cache_dir,
        fasta_path=args.fasta_path,
        head_id=args.head_id,
        batch_size=args.batch_size,
    )
    head_specs = data_module.head_specs
    head_names = tuple(spec.head_id for spec in head_specs)
    if len(head_names) != 1:
        raise ValueError(f"Torch eval currently expects one head, got {head_names}")
    head_name = head_names[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_strategy, precompute_standardized_convs = _split_torch_strategy(args.strategy)
    dtype_policy = DtypePolicy.full_float32()
    if base_strategy == "bf16_params" or base_strategy in TORCH_TRUE_QUANT_STRATEGIES:
        dtype_policy = DtypePolicy.aggressive_bfloat16()
    weights_path = args.torch_weights.expanduser().resolve()
    if weights_path.suffix in {".pt", ".pth", ".safetensors"}:
        payload = _load_torch_checkpoint_payload(weights_path, map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"Torch checkpoint lacks model_state_dict: {weights_path}")
        assay_type = payload.get("assay_type", "atac")
        resolutions = tuple(int(res) for res in payload.get("resolutions", (128,)))
        num_organisms = int(payload.get("num_organisms", 1))
        model = AlphaGenome(num_organisms=2, dtype_policy=dtype_policy)
        model = remove_all_heads(model)
        add_head(
            model,
            head_name,
            create_finetuning_head(
                assay_type,
                n_tracks=len(head_specs[0].tracks),
                resolutions=resolutions,
                num_organisms=num_organisms,
            ),
        )
        effective_torch_modules = tuple(payload.get("torch_effective_conv_modules") or ())
        if not effective_torch_modules:
            effective_jax_paths = tuple(str(path) for path in payload.get("backbone_effective_conv_paths", ()))
            effective_torch_modules = jax_effective_paths_to_torch(effective_jax_paths)
        materialize_effective_convs(model, effective_torch_modules)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
    else:
        model = AlphaGenome.from_pretrained(
            weights_path,
            dtype_policy=dtype_policy,
            device=device,
        )
    model.eval()
    standardized_conv_stats: dict[str, Any] = {
        "standardized_convs_materialized": 0,
        "standardized_conv_paths": [],
    }
    if precompute_standardized_convs:
        materialized = materialize_standardized_convs(model)
        standardized_conv_stats = {
            "standardized_convs_materialized": len(materialized),
            "standardized_conv_paths": list(materialized[:200]),
            "standardized_conv_paths_truncated": len(materialized) > 200,
        }
    quant_stats = apply_torch_quant_policy(model, base_strategy, nf4_block_size=args.nf4_block_size)
    quant_stats.update(standardized_conv_stats)
    quant_stats["strategy"] = args.strategy
    quant_stats["base_strategy"] = base_strategy
    quant_stats["precompute_standardized_convs"] = precompute_standardized_convs
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    split_results: dict[str, Any] = {}
    total_batches = 0
    total_examples = 0
    start_time = time.perf_counter()
    with torch.inference_mode():
        for split in args.splits.split(","):
            split = split.strip()
            if not split:
                continue
            preds: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            losses: list[float] = []
            batches = 0
            examples = 0
            for batch_np in data_module.iter_batches(split, shuffle=False):
                seq = torch.as_tensor(batch_np["sequences"], device=device, dtype=torch.float32)
                batch_examples = int(seq.shape[0])
                org = torch.zeros((seq.shape[0],), device=device, dtype=torch.long)
                target_np = _maybe_bin_128bp_np(np.asarray(batch_np[f"targets_{head_name}"], dtype=np.float32))
                target = torch.as_tensor(target_np, device=device, dtype=torch.float32)
                outputs = model(
                    seq,
                    org,
                    heads=(head_name,),
                    resolutions=(128,),
                    return_scaled_predictions=False,
                    channels_last=True,
                )
                pred = outputs[head_name][128].float()
                losses.append(float(F.mse_loss(pred, target).detach().cpu()))
                preds.append(pred.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())
                batches += 1
                total_batches += 1
                examples += batch_examples
                total_examples += batch_examples
                if args.max_batches and batches >= args.max_batches:
                    break
            if preds:
                pred_all = np.concatenate(preds, axis=0)
                target_all = np.concatenate(targets, axis=0)
                metrics = r2_metrics_np(pred_all, target_all)
                metrics["loss"] = float(np.mean(losses))
                metrics["batches"] = batches
                metrics["examples"] = examples
            else:
                metrics = {
                    "loss": float("nan"),
                    "batches": 0,
                    "examples": 0,
                    "r2_global": float("nan"),
                    "r2_over_loci": float("nan"),
                    "r2_over_cell_types": float("nan"),
                    "differential_pearson_r": float("nan"),
                }
            split_results[split] = {head_name: metrics}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = {
            "max_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "max_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        }
    return {
        "backend": "torch",
        "strategy": args.strategy,
        "checkpoint": str(args.torch_weights.expanduser().resolve()),
        "splits": split_results,
        "quantization": quant_stats,
        "elapsed_sec": elapsed,
        "batch_size": args.batch_size,
        "batches": total_batches,
        "batches_per_sec": total_batches / elapsed if elapsed > 0 else None,
        "examples": total_examples,
        "examples_per_sec": total_examples / elapsed if elapsed > 0 else None,
        "torch_cuda_memory": cuda_memory,
    }


def write_markdown(path: Path, metrics: Mapping[str, Any]) -> None:
    examples_per_sec = metrics.get("examples_per_sec")
    examples_per_sec_text = (
        f"{float(examples_per_sec):.4f}" if examples_per_sec is not None else "NA"
    )
    lines = [
        f"# Quant ablation: {metrics['backend']} {metrics['strategy']}",
        "",
        f"- checkpoint: `{metrics['checkpoint']}`",
        f"- batch_size: `{metrics.get('batch_size')}`",
        f"- elapsed_sec: `{metrics['elapsed_sec']:.2f}`",
        f"- examples_per_sec: `{examples_per_sec_text}`",
        f"- batches: `{metrics['batches']}`",
        f"- examples: `{metrics.get('examples')}`",
        f"- converted leaves: `{metrics['quantization'].get('converted')}`",
        f"- simulated storage: `{metrics['quantization'].get('simulated_storage')}`",
    ]
    torch_cuda = metrics.get("torch_cuda_memory")
    if isinstance(torch_cuda, dict):
        lines.extend(
            [
                f"- torch max allocated MiB: `{torch_cuda.get('max_allocated_mib'):.0f}`",
                f"- torch max reserved MiB: `{torch_cuda.get('max_reserved_mib'):.0f}`",
            ]
        )
    lines.extend(
        [
            "",
            "| split | head | loss | diff Pearson | r2_global | r2_over_loci | batches |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split, heads in metrics["splits"].items():
        for head, values in heads.items():
            lines.append(
                "| {split} | {head} | {loss:.4f} | {diff:.4f} | {r2:.4f} | {loci:.4f} | {batches} |".format(
                    split=split,
                    head=head,
                    loss=values.get("loss", float("nan")),
                    diff=values.get("differential_pearson_r", float("nan")),
                    r2=values.get("r2_global", float("nan")),
                    loci=values.get("r2_over_loci", float("nan")),
                    batches=values.get("batches", 0),
                )
            )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "torch"), default="jax")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--torch-repo", type=Path, default=Path(__file__).resolve().parents[2] / "alphagenome-pytorch")
    parser.add_argument("--torch-weights", type=Path, default=None)
    parser.add_argument("--base-checkpoint-path", type=Path, default=None)
    parser.add_argument("--model-version", default="all_folds")
    parser.add_argument("--bigwig-dir", type=Path, default=DEFAULT_BIGWIG_DIR)
    parser.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--head-id", default="humanbraindev_atac")
    parser.add_argument("--window-size", type=int, default=131072)
    parser.add_argument("--stride", type=int, default=131072)
    parser.add_argument("--valid-chroms", default="chr8")
    parser.add_argument("--test-chroms", default="chr9")
    parser.add_argument("--exclude-chroms", default="chrM,chrY")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-workers", type=int, default=8)
    parser.add_argument("--window-workers", type=int, default=4)
    parser.add_argument("--target-cache-dir", type=Path, default=None)
    parser.add_argument("--target-cache-dtype", default="float16")
    parser.add_argument("--build-target-cache", action="store_true")
    parser.add_argument("--target-cache-workers", type=int, default=None)
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--nf4-block-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpu_path = out / "gpu_samples.csv"
    with GpuSampler(gpu_path, interval_sec=args.gpu_sample_interval):
        if args.backend == "jax":
            metrics = evaluate_jax(args)
        else:
            if args.torch_weights is None:
                raise ValueError("--torch-weights is required with --backend torch")
            metrics = evaluate_torch(args)
    metrics["gpu"] = _gpu_summary(gpu_path)
    _json_dump(out / "metrics.json", metrics)
    write_markdown(out / "metrics.md", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
