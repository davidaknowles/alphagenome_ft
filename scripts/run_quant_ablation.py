#!/usr/bin/env python
"""Evaluate merged AlphaGenome checkpoints under eval-only quantization policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model

from alphagenome_ft import create_model_with_heads
from alphagenome_ft.finetune import (
    BigWigDataModule,
    WindowedTargetCache,
    load_targets_config,
    prepare_head_specs,
    register_predefined_heads,
    validate_head_specs,
)
from alphagenome_ft.finetune.data import prepare_batch
from alphagenome_ft.finetune.metrics import r2_metrics
from alphagenome_ft.finetune.train import _add_stats, _finalize_r2_stats, _r2_stats
from scripts.run_humanbraindev_finetune import (
    build_targets_config,
    discover_bigwigs,
    make_chromosome_split_intervals,
    parse_chrom_set,
)


DEFAULT_BIGWIG_DIR = Path(
    "/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs"
)
DEFAULT_FASTA = Path("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa")
DEFAULT_MERGED = Path("outputs/quant_ablation/merged_jax_default_lora_locon")

NF4_CODEBOOK = jnp.asarray(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=jnp.float32,
)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _keypath_to_str(path: tuple) -> str:
    return "/".join(str(getattr(key, "key", key)) for key in path)


def _is_float_array(value: Any) -> bool:
    return hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)


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


def _path_is_stem(path: str) -> bool:
    lowered = path.lower()
    return "conv1" in lowered or "dna_embedder" in lowered or "sequence_encoder/conv" in lowered


def _path_is_late_conv(path: str) -> bool:
    lowered = path.lower().replace(".", "_")
    return (
        ("down_blocks_4" in lowered or "down_blocks_5" in lowered)
        or ("down_res_block_4" in lowered or "down_res_block_5" in lowered)
        or ("down_blocks/4" in lowered or "down_blocks/5" in lowered)
    )


def _classify_leaf(path: str, value: Any) -> str:
    if not _is_float_array(value):
        return "other"
    if path.endswith("/w") and getattr(value, "ndim", 0) == 2:
        return "linear"
    if path.endswith("/w") and getattr(value, "ndim", 0) == 3:
        width = int(value.shape[0])
        return "conv1x1" if width == 1 else "conv"
    return "other"


def _policy_selects(path: str, value: Any, strategy: str) -> bool:
    if _path_is_sensitive(path):
        return False
    kind = _classify_leaf(path, value)
    if kind == "other":
        return False
    conservative = "conservative" in strategy
    aggressive = "aggressive" in strategy
    if "linear" in strategy:
        return kind == "linear" and (aggressive or value.size >= 262_144)
    if "1x1conv" in strategy:
        return kind == "linear" or kind == "conv1x1"
    if "late_conv" in strategy:
        return kind == "linear" or (kind == "conv" and _path_is_late_conv(path))
    if "all_conv" in strategy:
        return kind in {"linear", "conv1x1", "conv"} and not _path_is_stem(path)
    return False


def _nf4_roundtrip(value, *, block_size: int = 64):
    original_dtype = value.dtype
    flat = value.astype(jnp.float32).reshape(-1)
    if flat.size == 0:
        return value
    pad = (-flat.size) % block_size
    if pad:
        flat = jnp.pad(flat, (0, pad))
    blocks = flat.reshape((-1, block_size))
    scale = jnp.max(jnp.abs(blocks), axis=1, keepdims=True)
    scale = jnp.maximum(scale, jnp.asarray(1e-8, dtype=jnp.float32))
    normalized = jnp.clip(blocks / scale, -1.0, 1.0)
    distances = jnp.abs(normalized[..., None] - NF4_CODEBOOK[None, None, :])
    indices = jnp.argmin(distances, axis=-1)
    dequant = NF4_CODEBOOK[indices] * scale
    dequant = dequant.reshape(-1)[: value.size].reshape(value.shape)
    return dequant.astype(original_dtype)


def _fp8_roundtrip(value):
    original_dtype = value.dtype
    if not hasattr(jnp, "float8_e4m3fn"):
        raise RuntimeError("This JAX build does not expose jnp.float8_e4m3fn.")
    return value.astype(jnp.float8_e4m3fn).astype(original_dtype)


def apply_jax_quant_policy(params, strategy: str, *, nf4_block_size: int = 64):
    if strategy == "default":
        return params, {"strategy": strategy, "converted": 0, "simulated_storage": False}
    converted_paths: list[str] = []

    def convert(path_tuple, value):
        path = _keypath_to_str(path_tuple)
        if strategy == "bf16_params":
            if _is_float_array(value) and not _path_is_sensitive(path):
                converted_paths.append(path)
                return value.astype(jnp.bfloat16)
            return value
        if not _policy_selects(path, value, strategy):
            return value
        converted_paths.append(path)
        if strategy.startswith("nf4_"):
            return _nf4_roundtrip(value, block_size=nf4_block_size)
        if strategy.startswith("fp8_"):
            return _fp8_roundtrip(value)
        raise ValueError(f"Unknown quantization strategy: {strategy}")

    converted = jax.tree_util.tree_map_with_path(convert, params)
    return converted, {
        "strategy": strategy,
        "converted": len(converted_paths),
        "converted_paths": converted_paths[:200],
        "converted_paths_truncated": len(converted_paths) > 200,
        "simulated_storage": strategy.startswith(("nf4_", "fp8_")),
    }


def _torch_policy_selects(name: str, module: Any, strategy: str) -> bool:
    import torch.nn as nn

    lowered = name.lower()
    if any(token in lowered for token in ("heads", "norm", "embedding", "adapter", "lora_", "locon_")):
        return False
    if isinstance(module, nn.Linear):
        if "linear" in strategy:
            return True
        if "1x1conv" in strategy or "late_conv" in strategy or "all_conv" in strategy:
            return True
    if isinstance(module, nn.Conv1d):
        kernel = int(module.kernel_size[0])
        if "1x1conv" in strategy:
            return kernel == 1
        if "late_conv" in strategy:
            return kernel > 1 and (
                "down_blocks.4" in lowered
                or "down_blocks.5" in lowered
                or "down_res_block_4" in lowered
                or "down_res_block_5" in lowered
            )
        if "all_conv" in strategy:
            return kernel > 1 and "dna_embedder" not in lowered and "conv1" not in lowered
    return False


def _nf4_roundtrip_np(array: np.ndarray, *, block_size: int = 64) -> np.ndarray:
    codebook = np.asarray(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        dtype=np.float32,
    )
    original_shape = array.shape
    flat = array.astype(np.float32, copy=False).reshape(-1)
    pad = (-flat.size) % block_size
    if pad:
        flat = np.pad(flat, (0, pad))
    blocks = flat.reshape((-1, block_size))
    scale = np.maximum(np.max(np.abs(blocks), axis=1, keepdims=True), 1e-8)
    normalized = np.clip(blocks / scale, -1.0, 1.0)
    indices = np.argmin(np.abs(normalized[..., None] - codebook[None, None, :]), axis=-1)
    return (codebook[indices] * scale).reshape(-1)[: np.prod(original_shape)].reshape(original_shape)


def apply_torch_quant_policy(model: Any, strategy: str, *, nf4_block_size: int = 64) -> dict[str, Any]:
    if strategy == "default":
        return {"strategy": strategy, "converted": 0, "simulated_storage": False}
    converted: list[str] = []
    if strategy == "bf16_params":
        import torch

        model.to(dtype=torch.bfloat16)
        return {"strategy": strategy, "converted": -1, "simulated_storage": False}
    if strategy.startswith("nf4_"):
        import torch

        with torch.no_grad():
            for name, module in model.named_modules():
                if not _torch_policy_selects(name, module, strategy):
                    continue
                weight = getattr(module, "weight", None)
                if weight is None:
                    continue
                dequant = _nf4_roundtrip_np(weight.detach().float().cpu().numpy(), block_size=nf4_block_size)
                weight.copy_(torch.as_tensor(dequant, device=weight.device, dtype=weight.dtype))
                converted.append(f"{name}.weight")
        return {
            "strategy": strategy,
            "converted": len(converted),
            "converted_paths": converted[:200],
            "converted_paths_truncated": len(converted) > 200,
            "simulated_storage": True,
        }
    if strategy.startswith("fp8_"):
        from alphagenome_pytorch.low_precision import convert_linears_to_float8_training

        stats = convert_linears_to_float8_training(
            model,
            recipe="tensorwise",
            skip_name_patterns=("heads", "norm", "adapter", "lora_", "locon_"),
        )
        return {**stats.__dict__, "strategy": strategy, "simulated_storage": False}
    if strategy == "nvfp4_weight_only":
        from alphagenome_pytorch.low_precision import convert_linears_to_nvfp4_weight_only

        stats = convert_linears_to_nvfp4_weight_only(
            model,
            skip_name_patterns=("heads", "norm", "adapter", "lora_", "locon_"),
        )
        return {**stats.__dict__, "strategy": strategy, "simulated_storage": False}
    raise ValueError(f"Unsupported torch strategy: {strategy}")


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


def build_data_module(args: argparse.Namespace):
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
    model = create_model_with_heads(
        args.model_version,
        heads=tuple(head_names),
        checkpoint_path=args.base_checkpoint_path.expanduser().resolve()
        if args.base_checkpoint_path
        else None,
        detach_backbone=True,
        init_seq_len=args.window_size,
    )
    checkpoint_path = args.checkpoint.expanduser().resolve() / "checkpoint"
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(str(checkpoint_path), target=(model._params, model._state))
    if isinstance(restored, (tuple, list)) and len(restored) == 2:
        model._params, model._state = restored
    else:
        raise ValueError(f"Unexpected checkpoint payload from {checkpoint_path!s}")
    return model


def evaluate_jax(args: argparse.Namespace) -> dict[str, Any]:
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
    start_time = time.perf_counter()
    with model._device_context:
        for split in args.splits.split(","):
            split = split.strip()
            if not split:
                continue
            losses = {head: [] for head in head_names}
            stats_by_head: dict[str, dict[str, Any] | None] = {head: None for head in head_names}
            batches = 0
            for batch_np in data_module.iter_batches(split, shuffle=False):
                batch = prepare_batch(batch_np, organism_index_value, head_names)
                batch["strand_reindexing"] = strand_reindexing
                head_losses, head_stats = eval_step(model._params, model._state, batch)
                for head_name in head_names:
                    losses[head_name].append(float(np.asarray(head_losses[head_name])))
                    stats_by_head[head_name] = _add_stats(stats_by_head[head_name], head_stats[head_name])
                batches += 1
                total_batches += 1
                if args.max_batches and batches >= args.max_batches:
                    break
            split_results[split] = {}
            for head_name in head_names:
                metrics = {
                    "loss": float(np.mean(losses[head_name])) if losses[head_name] else float("nan"),
                    "batches": batches,
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
        "batches": total_batches,
        "batches_per_sec": total_batches / elapsed if elapsed > 0 else None,
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

    data_module, head_specs = build_data_module(args)
    head_names = tuple(spec.head_id for spec in head_specs)
    if len(head_names) != 1:
        raise ValueError(f"Torch eval currently expects one head, got {head_names}")
    head_name = head_names[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_policy = DtypePolicy.mixed_precision()
    if args.strategy == "bf16_params":
        dtype_policy = DtypePolicy.aggressive_bfloat16()
    model = AlphaGenome.from_pretrained(
        args.torch_weights.expanduser().resolve(),
        dtype_policy=dtype_policy,
        device=device,
    )
    model.eval()
    quant_stats = apply_torch_quant_policy(model, args.strategy, nf4_block_size=args.nf4_block_size)

    split_results: dict[str, Any] = {}
    total_batches = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for split in args.splits.split(","):
            split = split.strip()
            if not split:
                continue
            preds: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            losses: list[float] = []
            batches = 0
            for batch_np in data_module.iter_batches(split, shuffle=False):
                seq = torch.as_tensor(batch_np["sequences"], device=device, dtype=torch.float32)
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
                if args.max_batches and batches >= args.max_batches:
                    break
            if preds:
                pred_all = np.concatenate(preds, axis=0)
                target_all = np.concatenate(targets, axis=0)
                metrics = r2_metrics(pred_all, target_all)
                metrics["loss"] = float(np.mean(losses))
                metrics["batches"] = batches
            else:
                metrics = {
                    "loss": float("nan"),
                    "batches": 0,
                    "r2_global": float("nan"),
                    "r2_over_loci": float("nan"),
                    "r2_over_cell_types": float("nan"),
                    "differential_pearson_r": float("nan"),
                }
            split_results[split] = {head_name: metrics}
    elapsed = time.perf_counter() - start_time
    return {
        "backend": "torch",
        "strategy": args.strategy,
        "checkpoint": str(args.torch_weights.expanduser().resolve()),
        "splits": split_results,
        "quantization": quant_stats,
        "elapsed_sec": elapsed,
        "batches": total_batches,
        "batches_per_sec": total_batches / elapsed if elapsed > 0 else None,
    }


def write_markdown(path: Path, metrics: Mapping[str, Any]) -> None:
    lines = [
        f"# Quant ablation: {metrics['backend']} {metrics['strategy']}",
        "",
        f"- checkpoint: `{metrics['checkpoint']}`",
        f"- elapsed_sec: `{metrics['elapsed_sec']:.2f}`",
        f"- batches: `{metrics['batches']}`",
        f"- converted leaves: `{metrics['quantization'].get('converted')}`",
        f"- simulated storage: `{metrics['quantization'].get('simulated_storage')}`",
        "",
        "| split | head | loss | diff Pearson | r2_global | r2_over_loci | batches |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
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
