"""PyTorch backend adapter.

This adapter intentionally delegates model construction/training to the
``alphagenome-pytorch`` repository while this repo owns shared dataset discovery
and split generation.  That keeps the backend boundary narrow and preserves the
PyTorch implementation's low-precision options.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from alphagenome_ft.finetune.backends.base import PreparedRun


def _write_split_beds(intervals, split_dir: Path) -> dict[str, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, split_intervals in intervals.items():
        path = split_dir / f"{split}.bed"
        with path.open("w") as handle:
            for interval in split_intervals:
                handle.write(
                    f"{interval.chromosome}\t{int(interval.start)}\t{int(interval.end)}\n"
                )
        paths[str(split)] = path
    return paths


@dataclass(frozen=True)
class TorchBackendConfig:
    """Options required to invoke the PyTorch training backend."""

    repo_dir: Path
    pretrained_weights: Path
    output_dir: Path
    run_name: str | None
    mode: str
    dtype: str
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    lora_rank: int
    lora_alpha: int
    lora_targets: str
    fp8_recipe: str
    fp8_min_feature_multiple: int
    fp8_skip_name_patterns: str
    fp4_min_feature_multiple: int
    fp4_mode: str
    fp4_skip_name_patterns: str
    gradient_checkpointing: bool
    track_means_samples: int | None
    num_workers: int
    max_io_workers: int
    save_delta: bool
    save_checkpoints: bool
    wandb: bool
    wandb_project: str | None
    wandb_entity: str | None
    python_executable: Path | None = None


class TorchSubprocessBackend:
    """Backend that launches ``../alphagenome-pytorch/scripts/finetune.py``."""

    name = "torch"

    def __init__(self, config: TorchBackendConfig) -> None:
        self.config = config

    def _command(self, prepared: PreparedRun, bed_paths: dict[str, Path]) -> list[str]:
        cfg = self.config
        python = str(cfg.python_executable or Path(sys.executable))
        train_bed = bed_paths.get("train")
        val_bed = bed_paths.get("valid")
        if train_bed is None or val_bed is None:
            raise ValueError("Torch backend requires train and valid splits.")
        if not prepared.intervals.get("train"):
            raise ValueError("Torch backend requires at least one train interval.")

        cmd = [
            python,
            str(cfg.repo_dir / "scripts" / "finetune.py"),
            "--mode",
            cfg.mode,
            "--genome",
            str(prepared.fasta_path),
            "--modality",
            "atac",
            "--bigwig",
            *[str(path) for path in prepared.bigwigs],
            "--train-bed",
            str(train_bed),
            "--val-bed",
            str(val_bed),
            "--sequence-length",
            str(int(prepared.intervals["train"][0].end - prepared.intervals["train"][0].start)),
            "--resolutions",
            "1,128",
            "--pretrained-weights",
            str(cfg.pretrained_weights),
            "--output-dir",
            str(cfg.output_dir),
            "--epochs",
            str(cfg.epochs),
            "--batch-size",
            str(cfg.batch_size),
            "--gradient-accumulation-steps",
            str(cfg.gradient_accumulation_steps),
            "--lr",
            str(cfg.learning_rate),
            "--weight-decay",
            str(cfg.weight_decay),
            "--warmup-steps",
            str(cfg.warmup_steps),
            "--lora-rank",
            str(cfg.lora_rank),
            "--lora-alpha",
            str(int(cfg.lora_alpha)),
            "--lora-targets",
            cfg.lora_targets,
            "--dtype",
            cfg.dtype,
            "--fp8-recipe",
            cfg.fp8_recipe,
            "--fp8-min-feature-multiple",
            str(cfg.fp8_min_feature_multiple),
            "--fp8-skip-name-patterns",
            cfg.fp8_skip_name_patterns,
            "--fp4-min-feature-multiple",
            str(cfg.fp4_min_feature_multiple),
            "--fp4-mode",
            cfg.fp4_mode,
            "--fp4-skip-name-patterns",
            cfg.fp4_skip_name_patterns,
            "--num-workers",
            str(cfg.num_workers),
            "--max-io-workers",
            str(cfg.max_io_workers),
        ]
        if cfg.run_name:
            cmd.extend(["--run-name", cfg.run_name])
        if cfg.track_means_samples is not None:
            cmd.extend(["--track-means-samples", str(cfg.track_means_samples)])
        if cfg.gradient_checkpointing:
            cmd.append("--gradient-checkpointing")
        if cfg.save_delta:
            cmd.append("--save-delta")
        if not cfg.save_checkpoints:
            cmd.append("--no-save-checkpoints")
        if cfg.wandb:
            cmd.extend(["--wandb", "--wandb-project", cfg.wandb_project or "alphagenome-finetune"])
            if cfg.wandb_entity:
                cmd.extend(["--wandb-entity", cfg.wandb_entity])
        return cmd

    def run(self, prepared: PreparedRun) -> None:
        cfg = self.config
        repo_dir = cfg.repo_dir.expanduser().resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(f"Torch backend repo not found: {repo_dir}")
        finetune_script = repo_dir / "scripts" / "finetune.py"
        if not finetune_script.exists():
            raise FileNotFoundError(f"Torch finetune script not found: {finetune_script}")
        if not cfg.pretrained_weights.exists():
            raise FileNotFoundError(
                f"Torch pretrained weights not found: {cfg.pretrained_weights}"
            )

        split_dir = cfg.output_dir / "splits"
        bed_paths = _write_split_beds(prepared.intervals, split_dir)
        cmd = self._command(prepared, bed_paths)
        print("Torch backend command:")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=repo_dir, check=True)


__all__ = ["TorchBackendConfig", "TorchSubprocessBackend"]
