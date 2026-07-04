#!/usr/bin/env python
"""Prediction-only parity and VRAM checks for OG AlphaGenome low-VRAM strategies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_quant_ablation import (  # noqa: E402
    TORCH_STDCONV_EFFECTIVE_SUFFIX,
    apply_torch_quant_policy,
    _split_torch_strategy,
)
from scripts.torch_effective_conv import materialize_standardized_convs  # noqa: E402


DEFAULT_FASTA = Path("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa")
DEFAULT_OG_WEIGHTS = Path(
    "/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors"
)
DEFAULT_WEFF_WEIGHTS = Path("outputs/og_low_vram/alphagenome_og_bf16_weff.safetensors")

TABLE_STRATEGIES: dict[str, str] = {
    "default": "default",
    "bf16_params": "bf16_params",
    "triton_conv": "bf16_triton_conv_stdconv_effective",
    "flexattention": "bf16_triton_conv_flexattn_stdconv_effective",
    "flex_lowres_bias": "bf16_triton_conv_flexattn_lowresbias_stdconv_effective",
    "no_intermediates": "bf16_triton_conv_no_intermediates_stdconv_effective",
    "triton_pool": "bf16_triton_conv_no_intermediates_tritonpool_stdconv_effective",
    "fused_embedder": "bf16_triton_conv_no_intermediates_tritonpool_fusedembed_stdconv_effective",
    "fused_down0": "bf16_triton_conv_no_intermediates_tritonpool_fuseddown0_stdconv_effective",
    "fused_embedder_down0": (
        "bf16_triton_conv_no_intermediates_tritonpool_fusedembed_fuseddown0_stdconv_effective"
    ),
    "all_features": (
        "bf16_triton_conv_no_intermediates_tritonpool_fusedembed_fuseddown0_"
        "flexattn_lowresbias_stdconv_effective"
    ),
}


@dataclass(frozen=True)
class Interval:
    chrom: str
    start: int
    end: int


class GpuSampler:
    def __init__(self, path: Path, interval_sec: float = 0.25):
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


def _add_torch_repo(path: Path) -> None:
    path = path.expanduser().resolve()
    for entry in (path / "src", path):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


def _chrom_length(fasta_path: Path, chrom: str) -> int:
    fai = fasta_path.with_suffix(fasta_path.suffix + ".fai")
    with fai.open() as handle:
        for line in handle:
            name, length, *_ = line.rstrip("\n").split("\t")
            if name == chrom:
                return int(length)
    raise KeyError(f"{chrom} not found in {fai}")


def _make_intervals(fasta_path: Path, chrom: str, window_size: int, max_windows: int | None) -> list[Interval]:
    chrom_len = _chrom_length(fasta_path, chrom)
    intervals = [
        Interval(chrom, start, start + window_size)
        for start in range(0, chrom_len - window_size + 1, window_size)
    ]
    if max_windows is not None:
        intervals = intervals[:max_windows]
    if not intervals:
        raise ValueError(f"No full windows of size {window_size} fit on {chrom} ({chrom_len} bp).")
    return intervals


def _one_hot_dna(seq: str) -> np.ndarray:
    encoded = np.zeros((len(seq), 4), dtype=np.float32)
    lookup = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "a": 0,
        "c": 1,
        "g": 2,
        "t": 3,
    }
    for idx, base in enumerate(seq):
        channel = lookup.get(base)
        if channel is not None:
            encoded[idx, channel] = 1.0
    return encoded


def _iter_sequence_batches(
    *,
    fasta: Any,
    intervals: list[Interval],
    batch_size: int,
) -> Iterable[tuple[int, np.ndarray]]:
    for start_idx in range(0, len(intervals), batch_size):
        batch_intervals = intervals[start_idx : start_idx + batch_size]
        seqs = [_one_hot_dna(fasta[iv.chrom][iv.start : iv.end]) for iv in batch_intervals]
        yield start_idx, np.stack(seqs, axis=0)


def _load_weff_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = path.expanduser().resolve()
    metadata: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw = handle.metadata() or {}
    if "alphagenome_fp4_metadata" in raw:
        metadata = json.loads(raw["alphagenome_fp4_metadata"])
    else:
        sidecar = path.with_suffix(path.suffix + ".json")
        if sidecar.exists():
            with sidecar.open() as handle:
                metadata = json.load(handle)
    return load_file(str(path), device="cpu"), metadata


def _load_model(args: argparse.Namespace, strategy: str, device: Any):
    import torch
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.model import AlphaGenome

    base_strategy, needs_effective = _split_torch_strategy(strategy)
    dtype_policy = (
        DtypePolicy.aggressive_bfloat16()
        if strategy != "default"
        else DtypePolicy.full_float32()
    )
    if needs_effective:
        model = AlphaGenome(dtype_policy=dtype_policy)
        materialized = materialize_standardized_convs(model)
        state, checkpoint_metadata = _load_weff_state(args.bf16_weff_weights)
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected W_eff load result: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        model.to(device)
        load_stats = {
            "weights_path": str(args.bf16_weff_weights.expanduser().resolve()),
            "loaded_bf16_weff_checkpoint": True,
            "materialized_before_load": len(materialized),
            "checkpoint_metadata": checkpoint_metadata,
        }
    else:
        model = AlphaGenome.from_pretrained(
            args.og_weights.expanduser().resolve(),
            dtype_policy=dtype_policy,
            device=device,
        )
        load_stats = {
            "weights_path": str(args.og_weights.expanduser().resolve()),
            "loaded_bf16_weff_checkpoint": False,
        }
    model.eval()

    if needs_effective:
        quant_strategy = base_strategy
        standardized_stats = {
            "standardized_convs_materialized": int(load_stats["materialized_before_load"]),
            "standardized_conv_source": "checkpoint",
        }
    else:
        quant_strategy = strategy
        standardized_stats = {
            "standardized_convs_materialized": 0,
            "standardized_conv_source": "runtime",
        }
    quant_stats = apply_torch_quant_policy(model, quant_strategy)
    quant_stats.update(standardized_stats)
    quant_stats["strategy"] = strategy
    quant_stats["base_strategy"] = base_strategy
    quant_stats["precompute_standardized_convs"] = needs_effective
    return model, {**load_stats, "quantization": quant_stats}


def _predict_batch(model: Any, seq_np: np.ndarray, *, head: str, device: Any):
    import torch

    seq = torch.as_tensor(seq_np, device=device, dtype=torch.float32)
    org = torch.zeros((seq.shape[0],), device=device, dtype=torch.long)
    outputs = model(
        seq,
        org,
        heads=(head,),
        resolutions=(128,),
        return_scaled_predictions=False,
        channels_last=True,
    )
    return outputs[head][128].detach().float().cpu().numpy()


def _write_reference(
    *,
    args: argparse.Namespace,
    model: Any,
    intervals: list[Interval],
    device: Any,
) -> dict[str, Any]:
    from pyfaidx import Fasta

    fasta = Fasta(str(args.fasta_path.expanduser().resolve()), as_raw=True, sequence_always_upper=True)
    ref_path = args.reference_predictions.expanduser().resolve()
    ref_path.parent.mkdir(parents=True, exist_ok=True)

    pred_map = None
    batches = 0
    examples = 0
    start_time = time.perf_counter()
    for start_idx, seq_np in _iter_sequence_batches(fasta=fasta, intervals=intervals, batch_size=args.batch_size):
        pred = _predict_batch(model, seq_np, head=args.head, device=device)
        if pred_map is None:
            pred_map = np.lib.format.open_memmap(
                ref_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(intervals), pred.shape[1], pred.shape[2]),
            )
        pred_map[start_idx : start_idx + pred.shape[0]] = pred
        pred_map.flush()
        batches += 1
        examples += int(pred.shape[0])
    elapsed = time.perf_counter() - start_time
    if pred_map is None:
        raise RuntimeError("No predictions were written.")
    return {
        "reference_predictions": str(ref_path),
        "reference_shape": list(pred_map.shape),
        "batches": batches,
        "examples": examples,
        "elapsed_sec": elapsed,
        "examples_per_sec": examples / elapsed if elapsed > 0 else None,
    }


def _compare_to_reference(
    *,
    args: argparse.Namespace,
    model: Any,
    intervals: list[Interval],
    device: Any,
) -> dict[str, Any]:
    from pyfaidx import Fasta

    fasta = Fasta(str(args.fasta_path.expanduser().resolve()), as_raw=True, sequence_always_upper=True)
    reference = np.load(args.reference_predictions.expanduser().resolve(), mmap_mode="r")
    if reference.shape[0] != len(intervals):
        raise ValueError(f"Reference has {reference.shape[0]} intervals, expected {len(intervals)}")

    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    n_values = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    batches = 0
    examples = 0
    start_time = time.perf_counter()
    for start_idx, seq_np in _iter_sequence_batches(fasta=fasta, intervals=intervals, batch_size=args.batch_size):
        pred = _predict_batch(model, seq_np, head=args.head, device=device)
        ref = np.asarray(reference[start_idx : start_idx + pred.shape[0]], dtype=np.float32)
        diff = pred - ref
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
        sum_abs += float(np.abs(diff).sum(dtype=np.float64))
        sum_sq += float(np.square(diff, dtype=np.float64).sum())
        pred64 = pred.astype(np.float64, copy=False)
        ref64 = ref.astype(np.float64, copy=False)
        sum_x += float(pred64.sum())
        sum_y += float(ref64.sum())
        sum_xx += float(np.square(pred64).sum())
        sum_yy += float(np.square(ref64).sum())
        sum_xy += float((pred64 * ref64).sum())
        n_values += int(diff.size)
        batches += 1
        examples += int(pred.shape[0])
    elapsed = time.perf_counter() - start_time

    denom = math.sqrt(max(n_values * sum_xx - sum_x * sum_x, 0.0)) * math.sqrt(
        max(n_values * sum_yy - sum_y * sum_y, 0.0)
    )
    pearson = (n_values * sum_xy - sum_x * sum_y) / denom if denom > 0 else float("nan")
    return {
        "reference_predictions": str(args.reference_predictions.expanduser().resolve()),
        "reference_shape": list(reference.shape),
        "batches": batches,
        "examples": examples,
        "elapsed_sec": elapsed,
        "examples_per_sec": examples / elapsed if elapsed > 0 else None,
        "parity": {
            "max_abs": max_abs,
            "mean_abs": sum_abs / n_values,
            "rmse": math.sqrt(sum_sq / n_values),
            "pearson": pearson,
            "values": n_values,
        },
    }


def _write_metrics(out_dir: Path, metrics: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    parity = metrics.get("parity") or {}
    gpu = metrics.get("gpu") or {}
    torch_mem = metrics.get("torch_cuda_memory") or {}
    lines = [
        f"# OG low-VRAM parity: {metrics['strategy']}",
        "",
        f"- window_size: `{metrics['window_size']}`",
        f"- chrom: `{metrics['chrom']}`",
        f"- batch_size: `{metrics['batch_size']}`",
        f"- examples: `{metrics['examples']}`",
        f"- examples_per_sec: `{metrics.get('examples_per_sec')}`",
        f"- nvidia-smi max MiB: `{gpu.get('max_mem_mib')}`",
        f"- torch max allocated MiB: `{torch_mem.get('max_allocated_mib')}`",
        f"- torch max reserved MiB: `{torch_mem.get('max_reserved_mib')}`",
        f"- parity max_abs: `{parity.get('max_abs')}`",
        f"- parity rmse: `{parity.get('rmse')}`",
        f"- parity pearson: `{parity.get('pearson')}`",
    ]
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=tuple(TABLE_STRATEGIES))
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--og-weights", type=Path, default=DEFAULT_OG_WEIGHTS)
    parser.add_argument("--bf16-weff-weights", type=Path, default=DEFAULT_WEFF_WEIGHTS)
    parser.add_argument(
        "--torch-repo",
        type=Path,
        default=Path("/gpfs/commons/home/daknowles/projects/alphagenome-pytorch"),
    )
    parser.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--chrom", default="chr9")
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--head", default="atac")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _add_torch_repo(args.torch_repo)

    import torch

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    strategy = TABLE_STRATEGIES[args.strategy]
    intervals = _make_intervals(
        args.fasta_path.expanduser().resolve(),
        args.chrom,
        args.window_size,
        args.max_windows,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, load_stats = _load_model(args, strategy, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    gpu_path = out_dir / "gpu_samples.csv"
    with torch.inference_mode(), GpuSampler(gpu_path, interval_sec=args.gpu_sample_interval):
        if args.write_reference:
            run_metrics = _write_reference(args=args, model=model, intervals=intervals, device=device)
        else:
            run_metrics = _compare_to_reference(args=args, model=model, intervals=intervals, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch_cuda_memory = {
            "max_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "max_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        }
    else:
        torch_cuda_memory = None

    metrics = {
        "strategy_key": args.strategy,
        "strategy": strategy,
        "write_reference": bool(args.write_reference),
        "head": args.head,
        "chrom": args.chrom,
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "intervals": len(intervals),
        "device": str(device),
        **load_stats,
        **run_metrics,
        "torch_cuda_memory": torch_cuda_memory,
        "gpu": _gpu_summary(gpu_path),
    }
    _write_metrics(out_dir, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
