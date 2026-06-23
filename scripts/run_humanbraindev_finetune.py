#!/usr/bin/env python
"""Finetune an AlphaGenome ATAC head on human brain development BigWigs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from alphagenome_ft import (
    BackboneLoRAConfig,
    create_model_with_heads,
    lora,
    parse_lora_target_names,
    parameter_utils,
)
from alphagenome_ft.finetune import (
    BigWigDataModule,
    build_fasta_index,
    load_intervals_from_dataframe,
    load_targets_config,
    prepare_head_specs,
    prepare_intervals_from_fold,
    prepare_intervals_from_split,
    register_predefined_heads,
    train,
    validate_head_specs,
)


DEFAULT_BIGWIG_DIR = Path(
    "/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs"
)
DEFAULT_FASTA = Path("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/humanbraindev_atac_heads_only")


def _positive_int_or_none(value: str) -> int | None:
    if value.lower() in {"none", "null", "0"}:
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer, 0, or none.")
    return parsed


def discover_bigwigs(bigwig_dir: Path) -> list[Path]:
    bigwigs = sorted(bigwig_dir.expanduser().glob("*.bw"))
    if not bigwigs:
        raise FileNotFoundError(f"No .bw files found in {bigwig_dir}")
    return bigwigs


def build_targets_config(bigwigs: list[Path], head_id: str) -> dict:
    return {
        "heads": [
            {
                "id": head_id,
                "source": "predefined",
                "kind": "atac",
                "resolutions": [1, 128],
                "apply_squashing": False,
                "targets": [
                    {"path": str(path), "label": path.stem}
                    for path in bigwigs
                ],
            }
        ]
    }


def read_fai_chrom_sizes(fasta_path: Path) -> dict[str, int]:
    fai_path = Path(f"{fasta_path}.fai")
    if not fai_path.exists():
        build_fasta_index(fasta_path)

    chrom_sizes: dict[str, int] = {}
    with fai_path.open() as handle:
        for raw in handle:
            fields = raw.rstrip("\n").split("\t")
            if len(fields) >= 2:
                chrom_sizes[fields[0]] = int(fields[1])
    if not chrom_sizes:
        raise ValueError(f"No chromosome sizes found in {fai_path}")
    return chrom_sizes


def make_chromosome_split_intervals(
    fasta_path: Path,
    *,
    window_size: int,
    stride: int,
    valid_chroms: set[str],
    test_chroms: set[str],
    exclude_chroms: set[str],
    limit_train: int | None,
    limit_valid: int | None,
    limit_test: int | None,
) -> dict:
    chrom_sizes = read_fai_chrom_sizes(fasta_path)
    rows: list[tuple[str, int, int, str]] = []

    for chrom, chrom_size in chrom_sizes.items():
        if chrom in exclude_chroms or "_" in chrom or chrom.startswith("chrUn"):
            continue
        if chrom_size < window_size:
            continue
        split = "train"
        if chrom in valid_chroms:
            split = "valid"
        elif chrom in test_chroms:
            split = "test"

        for start in range(0, chrom_size - window_size + 1, stride):
            rows.append((chrom, start, start + window_size, split))

    intervals_df = pd.DataFrame(rows, columns=["chromosome", "start", "end", "split"])
    return load_intervals_from_dataframe(
        intervals_df,
        window_size=None,
        limit_train=limit_train,
        limit_valid=limit_valid,
        limit_test=limit_test,
    )


def parse_chrom_set(value: str) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heads-only AlphaGenome ATAC finetuning on human brain development BigWigs."
    )
    parser.add_argument("--bigwig-dir", type=Path, default=DEFAULT_BIGWIG_DIR)
    parser.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--head-id", default="humanbraindev_atac")
    parser.add_argument("--model-version", default="all_folds")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--split-source",
        choices=("chromosome", "fold", "bed"),
        default="chromosome",
        help="Use local chromosome holdouts, Borzoi fold intervals, or an explicit BED.",
    )
    parser.add_argument("--interval-bed", type=Path, default=None)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--window-size", type=int, default=131072)
    parser.add_argument("--stride", type=int, default=131072)
    parser.add_argument("--valid-chroms", default="chr8")
    parser.add_argument("--test-chroms", default="chr9")
    parser.add_argument("--exclude-chroms", default="chrM,chrY")
    parser.add_argument("--limit-train", type=_positive_int_or_none, default=None)
    parser.add_argument("--limit-valid", type=_positive_int_or_none, default=None)
    parser.add_argument("--limit-test", type=_positive_int_or_none, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--max-train-steps", type=_positive_int_or_none, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--fp8-lora", action="store_true")
    parser.add_argument("--base-param-dtype", default="float32")
    parser.add_argument("--lora-param-dtype", default="float32")
    parser.add_argument("--activation-dtype", default="bfloat16")
    parser.add_argument("--base-compute-dtype", default="bfloat16")
    parser.add_argument("--lora-compute-dtype", default=None)
    parser.add_argument(
        "--lora-targets",
        default="default",
        help="Comma-separated hk.Linear names to adapt, or 'default'.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bigwig_dir = args.bigwig_dir.expanduser().resolve()
    fasta_path = args.fasta_path.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if args.split_source == "bed" and args.interval_bed is None:
        raise ValueError("--interval-bed is required when --split-source=bed")

    bigwigs = discover_bigwigs(bigwig_dir)
    print(f"Discovered {len(bigwigs)} BigWig target tracks in {bigwig_dir}")

    targets_config = load_targets_config(build_targets_config(bigwigs, args.head_id))
    head_specs = prepare_head_specs(targets_config, organism="HOMO_SAPIENS")
    validate_head_specs(head_specs)
    register_predefined_heads(head_specs)

    if args.split_source == "fold":
        intervals = prepare_intervals_from_fold(
            fold=args.fold,
            window_size=args.window_size,
            organism="HOMO_SAPIENS",
            limit_train=args.limit_train,
            limit_valid=args.limit_valid,
            limit_test=args.limit_test,
        )
    elif args.split_source == "bed":
        intervals = prepare_intervals_from_split(
            bed_path=args.interval_bed.expanduser().resolve(),
            window_size=args.window_size,
            limit_train=args.limit_train,
            limit_valid=args.limit_valid,
            limit_test=args.limit_test,
        )
    else:
        intervals = make_chromosome_split_intervals(
            fasta_path,
            window_size=args.window_size,
            stride=args.stride,
            valid_chroms=parse_chrom_set(args.valid_chroms),
            test_chroms=parse_chrom_set(args.test_chroms),
            exclude_chroms=parse_chrom_set(args.exclude_chroms),
            limit_train=args.limit_train,
            limit_valid=args.limit_valid,
            limit_test=args.limit_test,
        )

    for split, split_intervals in intervals.items():
        print(f"{split}: {len(split_intervals)} interval(s)")

    data_module = BigWigDataModule(
        intervals=intervals,
        fasta_path=fasta_path,
        head_specs=head_specs,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        drop_last=args.drop_last,
    )

    head_ids = [spec.head_id for spec in head_specs]
    backbone_lora_config = None
    if args.backbone_lora:
        lora_compute_dtype = args.lora_compute_dtype
        if args.fp8_lora and lora_compute_dtype is None:
            lora_compute_dtype = "fp8"
        backbone_lora_config = BackboneLoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            fp8_enabled=args.fp8_lora,
            base_param_dtype=args.base_param_dtype,
            lora_param_dtype=args.lora_param_dtype,
            activation_dtype=args.activation_dtype,
            base_compute_dtype=args.base_compute_dtype,
            lora_compute_dtype=lora_compute_dtype,
            target_names=parse_lora_target_names(args.lora_targets),
        )
        print(
            "Backbone LoRA enabled: "
            f"rank={backbone_lora_config.rank}, "
            f"alpha={backbone_lora_config.alpha}, "
            f"base_param_dtype={backbone_lora_config.base_param_dtype}, "
            f"lora_param_dtype={backbone_lora_config.lora_param_dtype}, "
            f"activation_dtype={backbone_lora_config.activation_dtype}, "
            f"base_compute_dtype={backbone_lora_config.base_compute_dtype}, "
            f"lora_compute_dtype={backbone_lora_config.resolved_lora_compute_dtype()}, "
            f"targets={sorted(backbone_lora_config.normalized_target_names())}"
        )

    model = create_model_with_heads(
        args.model_version,
        heads=head_ids,
        checkpoint_path=args.checkpoint_path,
        detach_backbone=not args.backbone_lora,
        init_seq_len=args.window_size,
        backbone_lora_config=backbone_lora_config,
        runtime_backbone_param_dtype=args.base_param_dtype,
    )
    if args.backbone_lora:
        lora_paths = lora.get_lora_parameter_paths(model._params)
        total_params = parameter_utils.count_parameters(model._params)
        lora_params = lora.count_lora_parameters(model._params)
        print(f"LoRA adapter leaves: {len(lora_paths)}")
        print(f"LoRA adapter parameters: {lora_params:,} ({lora_params / total_params:.4%})")
        if not lora_paths:
            raise RuntimeError("Backbone LoRA was requested but no LoRA parameters were created.")

    use_wandb = args.wandb_project is not None
    train(
        model=model,
        data_module=data_module,
        head_specs=head_specs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        seed=args.seed,
        max_train_steps=args.max_train_steps,
        heads_only=True,
        train_lora=args.backbone_lora,
        checkpoint_dir=checkpoint_dir,
        organism="HOMO_SAPIENS",
        best_metric="valid_loss",
        best_metric_mode="min",
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        verbose=True,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_config={
            "bigwig_dir": str(bigwig_dir),
            "num_bigwigs": len(bigwigs),
            "fasta_path": str(fasta_path),
            "split_source": args.split_source,
            "window_size": args.window_size,
            "stride": args.stride,
            "backbone_lora": args.backbone_lora,
            "lora_rank": args.lora_rank if args.backbone_lora else None,
            "lora_alpha": args.lora_alpha if args.backbone_lora else None,
            "fp8_lora": args.fp8_lora if args.backbone_lora else None,
            "base_param_dtype": args.base_param_dtype if args.backbone_lora else None,
            "lora_param_dtype": args.lora_param_dtype if args.backbone_lora else None,
            "activation_dtype": args.activation_dtype if args.backbone_lora else None,
            "base_compute_dtype": args.base_compute_dtype if args.backbone_lora else None,
            "lora_compute_dtype": lora_compute_dtype if args.backbone_lora else None,
        },
        num_devices=args.num_devices,
    )


if __name__ == "__main__":
    main()
