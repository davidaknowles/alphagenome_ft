#!/usr/bin/env python
"""Run one backend/precision/adapter comparison cell and summarize results."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path


DEFAULT_BIGWIG_DIR = Path(
    "/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs"
)
DEFAULT_FASTA = Path("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa")
DEFAULT_JAX_CHECKPOINT = Path(
    "/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1"
)
DEFAULT_TORCH_WEIGHTS = Path(
    "/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors"
)


def _positive_int_or_none(value: str) -> int | None:
    if value.lower() in {"none", "null", "0"}:
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer, 0, or none.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jax", "torch"), required=True)
    parser.add_argument("--precision", choices=("default", "bf16", "nvfp8", "nvfp4"), required=True)
    parser.add_argument("--adapter-strategy", choices=("lora", "lora+locon"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--bigwig-dir", type=Path, default=DEFAULT_BIGWIG_DIR)
    parser.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--jax-checkpoint", type=Path, default=DEFAULT_JAX_CHECKPOINT)
    parser.add_argument("--torch-weights", type=Path, default=DEFAULT_TORCH_WEIGHTS)
    parser.add_argument("--target-cache-dir", type=Path, default=None)
    parser.add_argument("--target-cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--limit-train", type=_positive_int_or_none, default=None)
    parser.add_argument("--limit-valid", type=_positive_int_or_none, default=None)
    parser.add_argument("--limit-test", type=_positive_int_or_none, default=None)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--auto-batch-size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-batch-max-attempts", type=int, default=6)
    parser.add_argument("--auto-batch-vram-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Maximum epoch cap; benchmark jobs normally stop by early-stopping patience.",
    )
    parser.add_argument("--max-train-steps", type=_positive_int_or_none, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--best-metric", default="valid_differential_pearson_r")
    parser.add_argument("--best-metric-mode", choices=("min", "max"), default="max")
    parser.add_argument("--target-workers", type=int, default=8)
    parser.add_argument("--window-workers", type=int, default=4)
    parser.add_argument("--torch-num-workers", type=int, default=8)
    parser.add_argument("--torch-max-io-workers", type=int, default=16)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None)
    parser.add_argument("--wandb-job-type", default="precision-adapter-cell")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    return parser.parse_args()


def _sample_gpu(stop: threading.Event, samples: list[dict[str, float]]) -> None:
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            )
            util_s, mem_s = out.strip().splitlines()[0].split(",")[:2]
            samples.append(
                {
                    "t": time.time(),
                    "gpu_util": float(util_s.strip()),
                    "mem_mib": float(mem_s.strip()),
                }
            )
        except Exception:
            pass
        stop.wait(1.0)


def _gpu_total_mib() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def _run_command_with_gpu_sampling(
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
) -> tuple[subprocess.CompletedProcess, float, list[dict[str, float]]]:
    stop = threading.Event()
    samples: list[dict[str, float]] = []
    sampler = threading.Thread(target=_sample_gpu, args=(stop, samples), daemon=True)
    sampler.start()
    start = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    elapsed = time.time() - start
    stop.set()
    sampler.join(timeout=3)
    return proc, elapsed, samples


def _log_mentions_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="replace")
    oom_markers = (
        "CUDA out of memory",
        "torch.OutOfMemoryError",
        "RESOURCE_EXHAUSTED",
        "Out of memory",
        "out of memory",
    )
    return any(marker in text for marker in oom_markers)


def _parse_jax_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        return {}
    record = records[-1]
    metrics: dict[str, float] = {"train_loss": record.get("train_epoch_loss")}
    for split, heads in record.get("metrics", {}).items():
        head_metrics = next(iter(heads.values())) if heads else {}
        for key, value in head_metrics.items():
            metrics[f"{split}_{key}"] = value
    return metrics


def _parse_torch_metrics(run_dir: Path, run_name: str) -> dict[str, float | str]:
    path = run_dir / run_name / "epoch_log.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = rows[-1]
    metrics: dict[str, float | str] = {}
    for key, value in row.items():
        if key == "timestamp" or value == "":
            continue
        out_key = {"val_loss": "valid_loss"}.get(key, key)
        try:
            metrics[out_key] = float(value)
        except ValueError:
            metrics[out_key] = value
    if "valid_differential_pearson_r" not in metrics:
        best_name = metrics.get("best_metric_name")
        best_value = metrics.get("best_metric_value")
        if best_name == "differential_pearson_r" and isinstance(best_value, float):
            metrics["valid_differential_pearson_r"] = best_value
        else:
            matches = [
                value
                for key, value in metrics.items()
                if key.endswith("_128bp_differential_pearson_r") and isinstance(value, float)
            ]
            if len(matches) == 1:
                metrics["valid_differential_pearson_r"] = matches[0]
    return metrics


def _command(args: argparse.Namespace, run_dir: Path, run_name: str) -> list[str]:
    python = Path.home() / "venv" / "jax" / "bin" / "python"
    cmd = [
        str(python),
        "scripts/run_humanbraindev_finetune.py",
        "--backend",
        args.backend,
        "--precision",
        args.precision,
        "--adapter-strategy",
        args.adapter_strategy,
        "--backbone-lora",
        "--bigwig-dir",
        str(args.bigwig_dir),
        "--fasta-path",
        str(args.fasta_path),
        "--checkpoint-dir",
        str(run_dir),
        "--split-source",
        "chromosome",
        "--window-size",
        "131072",
        "--stride",
        "131072",
        "--batch-size",
        str(args.batch_size),
        "--num-epochs",
        str(args.num_epochs),
        "--best-metric",
        args.best_metric,
        "--best-metric-mode",
        args.best_metric_mode,
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--learning-rate",
        "1e-4",
        "--weight-decay",
        "0.1",
        "--lora-rank",
        "32",
        "--lora-alpha",
        "32",
        "--locon-rank",
        "4",
        "--locon-alpha",
        "1",
        "--progress-interval",
        "1",
        "--prefetch-batches",
        "2",
        "--target-workers",
        str(args.target_workers),
        "--window-workers",
        str(args.window_workers),
        "--target-cache-dtype",
        args.target_cache_dtype,
        "--no-shuffle",
    ]
    if args.target_cache_dir is not None:
        cmd.extend(["--target-cache-dir", str(args.target_cache_dir)])
    if args.limit_train is not None:
        cmd.extend(["--limit-train", str(args.limit_train)])
    if args.limit_valid is not None:
        cmd.extend(["--limit-valid", str(args.limit_valid)])
    if args.limit_test is not None:
        cmd.extend(["--limit-test", str(args.limit_test)])
    if args.backend == "jax":
        cmd.extend(["--checkpoint-path", str(args.jax_checkpoint), "--eval-splits", "train,valid,test"])
        if args.max_train_steps is not None:
            cmd.extend(["--max-train-steps", str(args.max_train_steps)])
    else:
        cmd.extend(
            [
                "--torch-python",
                str(Path.home() / "venv" / "torchfix" / "bin" / "python"),
                "--lora-targets",
                "q_proj,v_proj",
                "--torch-output-dir",
                str(run_dir),
                "--torch-run-name",
                run_name,
                "--torch-pretrained-weights",
                str(args.torch_weights),
                "--torch-num-workers",
                str(args.torch_num_workers),
                "--torch-max-io-workers",
                str(args.torch_max_io_workers),
                "--torch-track-means-samples",
                "1",
                "--torch-no-save-checkpoints",
            ]
        )
    if args.wandb_project:
        cmd.extend(
            [
                "--wandb-project",
                args.wandb_project,
                "--wandb-run-name",
                args.wandb_run_name or run_name,
            ]
        )
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_group:
            cmd.extend(["--wandb-group", args.wandb_group])
        if args.wandb_tags:
            cmd.extend(["--wandb-tags", args.wandb_tags])
        if args.wandb_job_type:
            cmd.extend(["--wandb-job-type", args.wandb_job_type])
        if args.wandb_mode:
            cmd.extend(["--wandb-mode", args.wandb_mode])
    return cmd


def _probe_args(args: argparse.Namespace, batch_size: int) -> argparse.Namespace:
    probe = copy.copy(args)
    probe.batch_size = batch_size
    probe.limit_train = max(batch_size, 1)
    probe.limit_valid = 1
    probe.limit_test = 1
    probe.num_epochs = 1
    probe.max_train_steps = 1 if probe.backend == "jax" else None
    probe.wandb_project = None
    probe.wandb_entity = None
    probe.wandb_run_name = None
    probe.wandb_group = None
    probe.wandb_tags = None
    probe.wandb_job_type = None
    probe.wandb_mode = None
    return probe


def _select_batch_size(args: argparse.Namespace, run_dir: Path, run_name: str, env: dict[str, str]) -> dict:
    total_mib = _gpu_total_mib()
    threshold_mib = (total_mib * args.auto_batch_vram_fraction) if total_mib else None
    attempts = []
    attempted_batches: set[int] = set()
    batch_size = max(1, int(args.batch_size))
    best_success: dict | None = None
    probe_root = run_dir / "_batch_probe"
    probe_root.mkdir(parents=True, exist_ok=True)

    for attempt_idx in range(1, args.auto_batch_max_attempts + 1):
        if batch_size in attempted_batches:
            break
        attempted_batches.add(batch_size)
        probe_dir = probe_root / f"batch_{batch_size}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_log = probe_dir / "run.log"
        print(
            f"[auto-batch] attempt {attempt_idx}: probing batch_size={batch_size}",
            flush=True,
        )
        probe_cmd = _command(_probe_args(args, batch_size), probe_dir, f"{run_name}_batch_probe_{batch_size}")
        proc, elapsed, samples = _run_command_with_gpu_sampling(probe_cmd, log_path=probe_log, env=env)
        max_mem_mib = max((sample["mem_mib"] for sample in samples), default=None)
        oom = proc.returncode != 0 and _log_mentions_oom(probe_log)
        attempt = {
            "attempt": attempt_idx,
            "batch_size": batch_size,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "max_mem_mib": max_mem_mib,
            "oom": oom,
            "log": str(probe_log),
        }
        attempts.append(attempt)
        mem_desc = f"{max_mem_mib:.0f} MiB" if max_mem_mib is not None else "unknown"
        print(
            "[auto-batch] "
            f"batch_size={batch_size} returncode={proc.returncode} oom={oom} "
            f"max_mem={mem_desc} elapsed={elapsed:.1f}s",
            flush=True,
        )

        if proc.returncode == 0:
            best_success = attempt
            if threshold_mib is None or max_mem_mib is None or max_mem_mib >= threshold_mib:
                break
            next_batch = max(batch_size + 1, int(math.ceil(batch_size * 1.5)))
            batch_size = next_batch
            continue

        if oom and best_success is not None:
            break
        if oom and batch_size > 1:
            batch_size = max(1, batch_size // 2)
            continue
        raise RuntimeError(
            "Batch-size probe failed for a non-OOM reason. "
            f"See {probe_log} (returncode={proc.returncode})."
        )

    if best_success is None:
        raise RuntimeError(f"No batch size succeeded after {len(attempts)} attempt(s): {attempts}")

    print(
        "[auto-batch] selected "
        f"batch_size={int(best_success['batch_size'])} "
        f"(threshold={threshold_mib:.0f} MiB)" if threshold_mib is not None else
        f"[auto-batch] selected batch_size={int(best_success['batch_size'])}",
        flush=True,
    )
    return {
        "selected_batch_size": int(best_success["batch_size"]),
        "gpu_total_mib": total_mib,
        "vram_threshold_mib": threshold_mib,
        "attempts": attempts,
    }


def main() -> None:
    args = parse_args()
    run_name = args.run_name or (
        f"{args.backend}_{args.precision}_{args.adapter_strategy.replace('+', '_')}"
    )
    run_dir = args.output_root.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    samples_path = run_dir / "gpu_samples.csv"
    summary_path = run_dir / "summary.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd()}:{env.get('PYTHONPATH', '')}"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    batch_selection = None
    selected_batch_size = args.batch_size
    if args.auto_batch_size:
        batch_selection = _select_batch_size(args, run_dir, run_name, env)
        selected_batch_size = int(batch_selection["selected_batch_size"])
        args = copy.copy(args)
        args.batch_size = selected_batch_size

    cmd = _command(args, run_dir, run_name)
    proc, elapsed, samples = _run_command_with_gpu_sampling(cmd, log_path=log_path, env=env)

    with samples_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t", "gpu_util", "mem_mib"])
        writer.writeheader()
        writer.writerows(samples)

    util = [sample["gpu_util"] for sample in samples]
    mem = [sample["mem_mib"] for sample in samples]
    nonzero_util = [value for value in util if value > 0]
    metrics = (
        _parse_jax_metrics(run_dir)
        if args.backend == "jax"
        else _parse_torch_metrics(run_dir, run_name)
    )
    summary = {
        "backend": args.backend,
        "precision": args.precision,
        "adapter_strategy": args.adapter_strategy,
        "batch_size": selected_batch_size,
        "auto_batch_size": args.auto_batch_size,
        "batch_selection": batch_selection,
        "run_name": run_name,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "avg_gpu_util_pct": sum(util) / len(util) if util else None,
        "avg_nonzero_gpu_util_pct": (
            sum(nonzero_util) / len(nonzero_util) if nonzero_util else None
        ),
        "max_gpu_util_pct": max(util) if util else None,
        "max_mem_mib": max(mem) if mem else None,
        "log": str(log_path),
        "gpu_samples": str(samples_path),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": (args.wandb_run_name or run_name) if args.wandb_project else None,
        "wandb_group": args.wandb_group,
        "wandb_tags": args.wandb_tags,
        "wandb_job_type": args.wandb_job_type if args.wandb_project else None,
        "wandb_mode": args.wandb_mode,
        **metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
