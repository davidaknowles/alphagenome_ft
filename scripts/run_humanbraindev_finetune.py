#!/usr/bin/env python
"""Finetune an AlphaGenome ATAC head on human brain development BigWigs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model

from alphagenome_ft import (
    BackboneLoConConfig,
    BackboneLoRAConfig,
    create_model_with_heads,
    load_checkpoint,
    lora,
    parse_locon_target_names,
    parse_lora_target_names,
    parameter_utils,
)
from alphagenome_ft.finetune import (
    BigWigDataModule,
    MultiSpeciesDataModule,
    PreparedRun,
    TorchBackendConfig,
    TorchSubprocessBackend,
    WindowedTargetCache,
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
DEFAULT_TORCH_REPO = Path(__file__).resolve().parents[2] / "alphagenome-pytorch"
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
                "targets": [{"path": str(path), "label": path.stem} for path in bigwigs],
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
    include_chroms: set[str] | None = None,
) -> dict:
    chrom_sizes = read_fai_chrom_sizes(fasta_path)
    rows: list[tuple[str, int, int, str]] = []

    for chrom, chrom_size in chrom_sizes.items():
        if chrom in exclude_chroms:
            continue
        if include_chroms is not None and chrom not in include_chroms:
            continue
        if include_chroms is None and ("_" in chrom or chrom.startswith("chrUn")):
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


def parse_csv_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def metadata_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            try:
                return max(1, int(slurm_cpus))
            except ValueError:
                pass
        return max(1, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heads-only AlphaGenome ATAC finetuning on human brain development BigWigs."
    )
    parser.add_argument("--bigwig-dir", type=Path, default=DEFAULT_BIGWIG_DIR)
    parser.add_argument(
        "--targets-config",
        type=Path,
        default=None,
        help="Optional multi-head YAML/JSON target config. Overrides --bigwig-dir discovery.",
    )
    parser.add_argument(
        "--species-config",
        type=Path,
        default=None,
        help="JSON config for balanced multi-species JAX training.",
    )
    parser.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA)
    parser.add_argument(
        "--organism",
        choices=("HOMO_SAPIENS", "MUS_MUSCULUS"),
        default="HOMO_SAPIENS",
        help="AlphaGenome organism embedding for a single-genome run.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--backend",
        choices=("jax", "torch"),
        default="jax",
        help="Training backend. The shared launcher owns data discovery and split generation.",
    )
    parser.add_argument(
        "--precision",
        choices=("default", "bf16", "nvfp8", "nvfp4", "nvfp8_linear", "nf4_linear"),
        default="default",
        help="Backend-neutral precision preset. Explicit backend flags still override this.",
    )
    parser.add_argument(
        "--adapter-strategy",
        choices=("lora", "lora+locon"),
        default="lora",
        help="Backend-neutral adapter strategy for backbone finetuning.",
    )
    parser.add_argument("--head-id", default="humanbraindev_atac")
    parser.add_argument("--model-version", default="all_folds")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Restore head and adapter parameters from a saved last/best checkpoint.",
    )
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
    parser.add_argument("--include-chroms", default="")
    parser.add_argument("--limit-train", type=_positive_int_or_none, default=None)
    parser.add_argument("--limit-valid", type=_positive_int_or_none, default=None)
    parser.add_argument("--limit-test", type=_positive_int_or_none, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--max-train-steps", type=_positive_int_or_none, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--double-centered-correlation-loss-weight",
        type=float,
        default=None,
        help="Override every head's optional one-minus double-centered Pearson loss weight.",
    )
    parser.add_argument("--backbone-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--fp8-lora", action="store_true")
    parser.add_argument("--fp4-lora", action="store_true")
    parser.add_argument("--locon-rank", type=int, default=4)
    parser.add_argument("--locon-alpha", type=float, default=1.0)
    parser.add_argument("--locon-param-dtype", default="float32")
    parser.add_argument("--locon-compute-dtype", default=None)
    parser.add_argument(
        "--locon-targets",
        default="default",
        help="Comma-separated JAX StandardizedConv1D path substrings to adapt, or 'default'.",
    )
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
    parser.add_argument("--best-metric", default="valid_loss")
    parser.add_argument("--best-metric-mode", choices=("min", "max"), default="min")
    parser.add_argument(
        "--eval-splits",
        default="train,valid,test",
        help="Comma-separated splits to evaluate after each epoch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--target-workers", type=int, default=0)
    parser.add_argument("--window-workers", type=int, default=None)
    parser.add_argument("--target-cache-dir", type=Path, default=None)
    parser.add_argument("--target-cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--build-target-cache", action="store_true")
    parser.add_argument("--build-target-cache-only", action="store_true")
    parser.add_argument("--overwrite-target-cache", action="store_true")
    parser.add_argument("--target-cache-workers", type=int, default=None)
    parser.add_argument("--profile-host-timing", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-tags",
        default=None,
        help="Comma-separated W&B tags shared across JAX and torch backends.",
    )
    parser.add_argument("--wandb-job-type", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)

    torch_group = parser.add_argument_group("Torch backend")
    torch_group.add_argument("--torch-repo", type=Path, default=DEFAULT_TORCH_REPO)
    torch_group.add_argument("--torch-python", type=Path, default=None)
    torch_group.add_argument("--torch-pretrained-weights", type=Path, default=DEFAULT_TORCH_WEIGHTS)
    torch_group.add_argument("--torch-output-dir", type=Path, default=None)
    torch_group.add_argument("--torch-run-name", default=None)
    torch_group.add_argument(
        "--torch-mode",
        choices=("linear-probe", "lora", "locon", "lora+locon", "full", "encoder-only"),
        default=None,
        help="Override PyTorch training mode. Defaults to lora when --backbone-lora is set.",
    )
    torch_group.add_argument(
        "--torch-dtype",
        choices=(
            "bfloat16",
            "float32",
            "float16",
            "bfloat16-params",
            "float16-params",
            "nvfp8",
            "nvfp4",
            "nf4",
        ),
        default=None,
        help="PyTorch precision preset. Defaults are inferred from --fp4-lora/--fp8-lora.",
    )
    torch_group.add_argument("--torch-gradient-accumulation-steps", type=int, default=1)
    torch_group.add_argument("--torch-warmup-steps", type=int, default=500)
    torch_group.add_argument("--torch-locon-rank", type=int, default=4)
    torch_group.add_argument("--torch-locon-alpha", type=int, default=1)
    torch_group.add_argument(
        "--torch-locon-targets",
        default="down_blocks.4,down_blocks.5",
        help="Comma-separated Conv1d module substrings for PyTorch Locon modes.",
    )
    torch_group.add_argument(
        "--torch-fp8-recipe",
        choices=("tensorwise", "rowwise", "rowwise_with_gw_hp"),
        default="tensorwise",
    )
    torch_group.add_argument("--torch-fp8-min-feature-multiple", type=int, default=16)
    torch_group.add_argument(
        "--torch-fp8-skip-name-patterns",
        default="heads,original_layer,lora_,locon_,ia3,adapter",
    )
    torch_group.add_argument("--torch-fp8-include-name-patterns", default="")
    torch_group.add_argument("--torch-fp4-min-feature-multiple", type=int, default=16)
    torch_group.add_argument(
        "--torch-fp4-mode",
        choices=("qat", "weight-only"),
        default="weight-only",
    )
    torch_group.add_argument(
        "--torch-fp4-skip-name-patterns",
        default="heads,lora_,locon_,ia3,adapter",
    )
    torch_group.add_argument("--torch-fp4-include-name-patterns", default="")
    torch_group.add_argument("--torch-nf4-min-feature-multiple", type=int, default=16)
    torch_group.add_argument(
        "--torch-nf4-skip-name-patterns",
        default="heads,original_layer,lora_,locon_,ia3,adapter",
    )
    torch_group.add_argument("--torch-nf4-include-name-patterns", default="")
    torch_group.add_argument("--torch-gradient-checkpointing", action="store_true")
    torch_group.add_argument(
        "--torch-track-means-samples", type=_positive_int_or_none, default=None
    )
    torch_group.add_argument("--torch-num-workers", type=int, default=4)
    torch_group.add_argument("--torch-max-io-workers", type=int, default=16)
    torch_group.add_argument("--torch-save-delta", action="store_true", default=True)
    torch_group.add_argument("--torch-no-save-delta", action="store_false", dest="torch_save_delta")
    torch_group.add_argument("--torch-no-save-checkpoints", action="store_true")
    return parser.parse_args()


def _infer_torch_dtype(args: argparse.Namespace) -> str:
    if args.torch_dtype is not None:
        return args.torch_dtype
    if args.precision in {"nvfp8", "nvfp8_linear"}:
        return "nvfp8"
    if args.precision == "nvfp4":
        return "nvfp4"
    if args.precision == "nf4_linear":
        return "nf4"
    if args.precision == "bf16":
        return "bfloat16"
    lowp_tokens = {
        str(args.base_param_dtype).lower(),
        str(args.base_compute_dtype).lower(),
        str(args.lora_param_dtype).lower(),
        str(args.lora_compute_dtype).lower(),
    }
    if args.fp4_lora or "fp4" in lowp_tokens or "nvfp4" in lowp_tokens:
        return "nvfp4"
    if args.fp8_lora or "fp8" in lowp_tokens or "nvfp8" in lowp_tokens:
        return "nvfp8"
    if str(args.activation_dtype).lower() in {"float16", "fp16"}:
        return "float16"
    return "bfloat16"


def _infer_torch_mode(args: argparse.Namespace) -> str:
    if args.torch_mode is not None:
        return args.torch_mode
    if args.adapter_strategy == "lora+locon":
        return "lora+locon"
    return "lora" if args.backbone_lora else "linear-probe"


def _apply_precision_preset(args: argparse.Namespace) -> None:
    if args.precision in {"nvfp8", "nvfp8_linear"}:
        if args.fp4_lora:
            args.fp4_lora = False
        if args.base_param_dtype == "float32":
            args.base_param_dtype = "fp8"
        if args.lora_param_dtype == "float32":
            args.lora_param_dtype = "bfloat16"
        if args.locon_param_dtype == "float32":
            args.locon_param_dtype = "bfloat16"
        if args.lora_compute_dtype is None:
            args.lora_compute_dtype = "bfloat16"
        if args.precision == "nvfp8_linear" and not args.torch_fp8_include_name_patterns:
            args.torch_fp8_include_name_patterns = "tower"
    elif args.precision == "nvfp4":
        if args.fp8_lora:
            args.fp8_lora = False
        if args.base_param_dtype == "float32":
            args.base_param_dtype = "fp4"
        if args.lora_param_dtype == "float32":
            args.lora_param_dtype = "bfloat16"
        if args.locon_param_dtype == "float32":
            args.locon_param_dtype = "bfloat16"
        if args.lora_compute_dtype is None:
            args.lora_compute_dtype = "bfloat16"
    elif args.precision == "nf4_linear":
        if args.base_param_dtype == "float32":
            args.base_param_dtype = "bfloat16"
        if args.lora_param_dtype == "float32":
            args.lora_param_dtype = "bfloat16"
        if args.locon_param_dtype == "float32":
            args.locon_param_dtype = "bfloat16"
        if args.lora_compute_dtype is None:
            args.lora_compute_dtype = "bfloat16"
        if not args.torch_nf4_include_name_patterns:
            args.torch_nf4_include_name_patterns = "tower"
    elif args.precision == "bf16":
        if args.base_param_dtype == "float32":
            args.base_param_dtype = "bfloat16"
        if args.lora_param_dtype == "float32":
            args.lora_param_dtype = "bfloat16"
        if args.locon_param_dtype == "float32":
            args.locon_param_dtype = "bfloat16"


def main() -> None:
    args = parse_args()
    _apply_precision_preset(args)
    if args.backend == "jax" and args.precision in {"nvfp8_linear", "nf4_linear"}:
        raise ValueError(
            f"JAX precision preset {args.precision!r} is not implemented as a real "
            "transformer-linears-only quantized path. Use the torch backend for this preset."
        )

    bigwig_dir = args.bigwig_dir.expanduser().resolve()
    fasta_path = args.fasta_path.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    species_entries = None
    if args.species_config is not None:
        if args.targets_config is not None:
            raise ValueError("--species-config and --targets-config are mutually exclusive.")
        if args.backend != "jax":
            raise ValueError("Multi-species training currently requires --backend=jax.")
        species_config_path = args.species_config.expanduser().resolve()
        species_payload = json.loads(species_config_path.read_text())
        species_entries = species_payload.get("species")
        if not species_entries or len(species_entries) < 2:
            raise ValueError("Species config must contain at least two species entries.")
        config_root = species_config_path.parent
        normalized_entries = []
        for entry in species_entries:
            normalized = dict(entry)
            for key in ("fasta", "targets_config"):
                path = Path(normalized[key]).expanduser()
                if not path.is_absolute():
                    path = config_root / path
                normalized[key] = path.resolve()
            normalized_entries.append(normalized)
        species_entries = normalized_entries
        fasta_path = species_entries[0]["fasta"]
        targets_config_path = species_entries[0]["targets_config"]
        bigwig_dir = targets_config_path.parent
        targets_config = load_targets_config(targets_config_path)
        bigwigs = [
            Path(target["path"])
            for head in targets_config.get("heads", ())
            for target in head.get("targets", ())
        ]
        print(f"Loaded joint training configuration for {len(species_entries)} species.")
    elif args.targets_config is None:
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        bigwigs = discover_bigwigs(bigwig_dir)
        print(f"Discovered {len(bigwigs)} BigWig target tracks in {bigwig_dir}")
        targets_config = load_targets_config(build_targets_config(bigwigs, args.head_id))
    else:
        if args.backend != "jax":
            raise ValueError("Multi-head --targets-config runs currently require --backend=jax.")
        targets_config_path = args.targets_config.expanduser().resolve()
        bigwig_dir = targets_config_path.parent
        targets_config = load_targets_config(targets_config_path)
        bigwigs = [
            Path(target["path"])
            for head in targets_config.get("heads", ())
            for target in head.get("targets", ())
        ]
        print(
            f"Loaded {len(targets_config.get('heads', ()))} heads and "
            f"{len(bigwigs)} target tracks from {targets_config_path}"
        )

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if args.split_source == "bed" and args.interval_bed is None:
        raise ValueError("--interval-bed is required when --split-source=bed")
    head_specs = prepare_head_specs(
        targets_config,
        organism=None if species_entries is not None else args.organism,
    )
    if args.double_centered_correlation_loss_weight is not None:
        if args.double_centered_correlation_loss_weight < 0:
            raise ValueError("Double-centered correlation loss weight must be non-negative.")
        head_specs = [
            dataclasses.replace(
                spec,
                double_centered_correlation_loss_weight=(
                    args.double_centered_correlation_loss_weight
                ),
            )
            for spec in head_specs
        ]
    validate_head_specs(head_specs)
    register_predefined_heads(head_specs)

    if species_entries is not None:
        species_modules = {}
        organism_indices = {}
        expected_signature = [
            (
                spec.head_id,
                spec.kind,
                len(spec.tracks),
                tuple(track.name for track in spec.tracks),
                spec.loss_weight,
                spec.gene_loss_weight,
                spec.coverage_loss_weight,
                spec.double_centered_correlation_loss_weight,
            )
            for spec in head_specs
        ]
        for entry in species_entries:
            species_name = str(entry["name"])
            species_organism = str(entry.get("organism", args.organism))
            organism_enum = getattr(ag_dna_model.Organism, species_organism)
            organism_indices[species_name] = int(
                research_dna_model.convert_to_organism_index(organism_enum)
            )
            species_fasta = entry["fasta"]
            species_specs = prepare_head_specs(
                load_targets_config(entry["targets_config"]), organism=None
            )
            if args.double_centered_correlation_loss_weight is not None:
                species_specs = [
                    dataclasses.replace(
                        spec,
                        double_centered_correlation_loss_weight=(
                            args.double_centered_correlation_loss_weight
                        ),
                    )
                    for spec in species_specs
                ]
            validate_head_specs(species_specs)
            signature = [
                (
                    spec.head_id,
                    spec.kind,
                    len(spec.tracks),
                    tuple(track.name for track in spec.tracks),
                    spec.loss_weight,
                    spec.gene_loss_weight,
                    spec.coverage_loss_weight,
                    spec.double_centered_correlation_loss_weight,
                )
                for spec in species_specs
            ]
            if signature != expected_signature:
                raise ValueError(
                    f"Species {species_name} target heads do not match the shared head layout."
                )
            species_intervals = make_chromosome_split_intervals(
                species_fasta,
                window_size=args.window_size,
                stride=args.stride,
                valid_chroms=parse_chrom_set(str(entry["valid_chroms"])),
                test_chroms=parse_chrom_set(str(entry["test_chroms"])),
                exclude_chroms=parse_chrom_set(str(entry.get("exclude_chroms", ""))),
                limit_train=args.limit_train,
                limit_valid=args.limit_valid,
                limit_test=args.limit_test,
                include_chroms=parse_chrom_set(str(entry.get("include_chroms", ""))) or None,
            )
            species_modules[species_name] = BigWigDataModule(
                intervals=species_intervals,
                fasta_path=species_fasta,
                head_specs=species_specs,
                batch_size=args.batch_size,
                shuffle=not args.no_shuffle,
                drop_last=args.drop_last,
                target_workers=args.target_workers,
                window_workers=(
                    args.window_workers
                    if args.window_workers is not None
                    else min(args.batch_size, _available_cpu_count())
                ),
                target_cache_dir=None,
                target_cache_dtype=args.target_cache_dtype,
            )
            print(
                f"{species_name}: "
                + ", ".join(
                    f"{split}={len(windows)}" for split, windows in species_intervals.items()
                )
            )
        data_module = MultiSpeciesDataModule(
            species_modules,
            organism_indices=organism_indices,
        )
        intervals = data_module._intervals
    elif args.split_source == "fold":
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
            include_chroms=parse_chrom_set(args.include_chroms) or None,
        )

    if species_entries is None:
        for split, split_intervals in intervals.items():
            print(f"{split}: {len(split_intervals)} interval(s)")

    target_cache_dir = (
        args.target_cache_dir.expanduser().resolve() if args.target_cache_dir is not None else None
    )
    if species_entries is not None and target_cache_dir is not None:
        raise ValueError("Target caches are not yet supported with --species-config.")
    if args.build_target_cache:
        if target_cache_dir is None:
            raise ValueError("--target-cache-dir is required with --build-target-cache.")
        filtered_intervals = BigWigDataModule._filter_intervals_by_bigwig_chromosomes(
            intervals, head_specs
        )
        WindowedTargetCache.build(
            target_cache_dir,
            intervals=filtered_intervals,
            head_specs=head_specs,
            dtype=args.target_cache_dtype,
            workers=args.target_cache_workers or _available_cpu_count(),
            overwrite=args.overwrite_target_cache,
        )
        if args.build_target_cache_only:
            print(f"Target cache build complete: {target_cache_dir}")
            return
    elif args.build_target_cache_only:
        raise ValueError("--build-target-cache-only requires --build-target-cache.")

    if args.backend == "torch":
        torch_intervals = BigWigDataModule._filter_intervals_by_bigwig_chromosomes(
            intervals,
            head_specs,
        )
        torch_output_dir = (
            args.torch_output_dir.expanduser().resolve()
            if args.torch_output_dir is not None
            else checkpoint_dir
        )
        torch_python = args.torch_python
        if torch_python is None:
            default_torch_python = Path.home() / "venv" / "torchfix" / "bin" / "python"
            if default_torch_python.exists():
                torch_python = default_torch_python
        backend = TorchSubprocessBackend(
            TorchBackendConfig(
                repo_dir=args.torch_repo.expanduser().resolve(),
                pretrained_weights=args.torch_pretrained_weights.expanduser().resolve(),
                output_dir=torch_output_dir,
                run_name=args.torch_run_name or args.wandb_run_name,
                mode=_infer_torch_mode(args),
                dtype=_infer_torch_dtype(args),
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.torch_gradient_accumulation_steps,
                epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_steps=args.torch_warmup_steps,
                lora_rank=args.lora_rank,
                lora_alpha=int(args.lora_alpha),
                lora_targets=(
                    "q_proj,v_proj" if args.lora_targets == "default" else args.lora_targets
                ),
                locon_rank=args.torch_locon_rank or args.locon_rank,
                locon_alpha=args.torch_locon_alpha,
                locon_targets=(
                    "down_blocks.4,down_blocks.5"
                    if args.torch_locon_targets == "default"
                    else args.torch_locon_targets
                ),
                fp8_recipe=args.torch_fp8_recipe,
                fp8_min_feature_multiple=args.torch_fp8_min_feature_multiple,
                fp8_skip_name_patterns=args.torch_fp8_skip_name_patterns,
                fp8_include_name_patterns=args.torch_fp8_include_name_patterns,
                fp4_min_feature_multiple=args.torch_fp4_min_feature_multiple,
                fp4_mode=args.torch_fp4_mode,
                fp4_skip_name_patterns=args.torch_fp4_skip_name_patterns,
                fp4_include_name_patterns=args.torch_fp4_include_name_patterns,
                nf4_min_feature_multiple=args.torch_nf4_min_feature_multiple,
                nf4_skip_name_patterns=args.torch_nf4_skip_name_patterns,
                nf4_include_name_patterns=args.torch_nf4_include_name_patterns,
                gradient_checkpointing=args.torch_gradient_checkpointing,
                track_means_samples=args.torch_track_means_samples,
                num_workers=args.torch_num_workers,
                max_io_workers=args.torch_max_io_workers,
                save_delta=args.torch_save_delta,
                save_checkpoints=not args.torch_no_save_checkpoints,
                best_metric=args.best_metric,
                best_metric_mode=args.best_metric_mode,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                wandb=args.wandb_project is not None,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_run_name=args.wandb_run_name,
                wandb_group=args.wandb_group,
                wandb_tags=parse_csv_list(args.wandb_tags),
                wandb_job_type=args.wandb_job_type,
                wandb_mode=args.wandb_mode,
                run_metadata=(
                    ("backend", args.backend),
                    ("precision", args.precision),
                    ("adapter_strategy", args.adapter_strategy),
                    ("backbone_lora", metadata_value(args.backbone_lora)),
                    ("base_param_dtype", metadata_value(args.base_param_dtype)),
                    ("base_compute_dtype", metadata_value(args.base_compute_dtype)),
                    ("lora_param_dtype", metadata_value(args.lora_param_dtype)),
                    ("locon_param_dtype", metadata_value(args.locon_param_dtype)),
                    ("lora_compute_dtype", metadata_value(args.lora_compute_dtype)),
                    ("activation_dtype", metadata_value(args.activation_dtype)),
                    ("torch_dtype", metadata_value(_infer_torch_dtype(args))),
                    (
                        "torch_fp8_include_name_patterns",
                        metadata_value(args.torch_fp8_include_name_patterns),
                    ),
                    (
                        "torch_nf4_include_name_patterns",
                        metadata_value(args.torch_nf4_include_name_patterns),
                    ),
                    ("bigwig_dir", str(bigwig_dir)),
                    ("num_bigwigs", metadata_value(len(bigwigs))),
                    ("split_source", args.split_source),
                    ("window_size", metadata_value(args.window_size)),
                    ("stride", metadata_value(args.stride)),
                ),
                # Keep the venv path itself. Resolving follows the venv's python
                # symlink to the system interpreter and loses site-packages.
                python_executable=torch_python.expanduser() if torch_python else None,
            )
        )
        backend.run(
            PreparedRun(
                bigwig_dir=bigwig_dir,
                bigwigs=bigwigs,
                fasta_path=fasta_path,
                intervals=torch_intervals,
                head_specs=head_specs,
            )
        )
        return

    if species_entries is None:
        data_module = BigWigDataModule(
            intervals=intervals,
            fasta_path=fasta_path,
            head_specs=head_specs,
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle,
            drop_last=args.drop_last,
            target_workers=args.target_workers,
            window_workers=(
                args.window_workers
                if args.window_workers is not None
                else min(args.batch_size, _available_cpu_count())
            ),
            target_cache_dir=target_cache_dir,
            target_cache_dtype=args.target_cache_dtype,
        )

    head_ids = [spec.head_id for spec in head_specs]
    backbone_lora_config = None
    backbone_locon_config = None
    if args.backbone_lora:
        if args.fp8_lora and args.fp4_lora:
            raise ValueError("--fp8-lora and --fp4-lora are mutually exclusive.")
        lora_compute_dtype = args.lora_compute_dtype
        if lora_compute_dtype is None:
            if args.fp4_lora:
                lora_compute_dtype = "fp4"
            elif args.fp8_lora:
                lora_compute_dtype = "fp8"
        backbone_lora_config = BackboneLoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            fp8_enabled=args.fp8_lora,
            fp4_enabled=args.fp4_lora,
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
            f"fp8_enabled={backbone_lora_config.fp8_enabled}, "
            f"fp4_enabled={backbone_lora_config.fp4_enabled}, "
            f"base_param_dtype={backbone_lora_config.base_param_dtype}, "
            f"lora_param_dtype={backbone_lora_config.lora_param_dtype}, "
            f"activation_dtype={backbone_lora_config.activation_dtype}, "
            f"base_compute_dtype={backbone_lora_config.base_compute_dtype}, "
            f"lora_compute_dtype={backbone_lora_config.resolved_lora_compute_dtype()}, "
            f"targets={sorted(backbone_lora_config.normalized_target_names())}"
        )
        if args.adapter_strategy == "lora+locon":
            locon_compute_dtype = args.locon_compute_dtype or args.activation_dtype
            backbone_locon_config = BackboneLoConConfig(
                rank=args.locon_rank,
                alpha=args.locon_alpha,
                param_dtype=args.locon_param_dtype,
                compute_dtype=locon_compute_dtype,
                target_names=parse_locon_target_names(args.locon_targets),
            )
            print(
                "Backbone LoCon enabled: "
                f"rank={backbone_locon_config.rank}, "
                f"alpha={backbone_locon_config.alpha}, "
                f"param_dtype={backbone_locon_config.param_dtype}, "
                f"compute_dtype={backbone_locon_config.compute_dtype}, "
                f"targets={sorted(backbone_locon_config.normalized_target_names())}"
            )

    start_epoch = 1
    initial_global_step = 0
    initial_optimizer_state_path = None
    if args.resume_from is None:
        model = create_model_with_heads(
            args.model_version,
            heads=head_ids,
            checkpoint_path=args.checkpoint_path,
            detach_backbone=not args.backbone_lora,
            init_seq_len=args.window_size,
            backbone_lora_config=backbone_lora_config,
            backbone_locon_config=backbone_locon_config,
            runtime_backbone_param_dtype=args.base_param_dtype,
        )
    else:
        resume_from = args.resume_from.expanduser().resolve()
        resume_metrics_path = resume_from / "metrics.json"
        if not resume_metrics_path.exists():
            raise FileNotFoundError(f"Resume checkpoint metrics not found: {resume_metrics_path}")
        resume_record = json.loads(resume_metrics_path.read_text())
        start_epoch = int(resume_record["epoch"]) + 1
        initial_global_step = int(resume_record["global_step"])
        optimizer_state_path = resume_from / "optimizer_state"
        if optimizer_state_path.exists():
            initial_optimizer_state_path = optimizer_state_path
        model = load_checkpoint(
            resume_from,
            base_model_version=args.model_version,
            base_checkpoint_path=args.checkpoint_path,
            init_seq_len=args.window_size,
            backbone_lora_config=backbone_lora_config,
            backbone_locon_config=backbone_locon_config,
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
        organism=args.organism,
        best_metric=args.best_metric,
        best_metric_mode=args.best_metric_mode,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        verbose=True,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_tags=parse_csv_list(args.wandb_tags),
        wandb_job_type=args.wandb_job_type,
        wandb_mode=args.wandb_mode,
        wandb_config={
            "backend": args.backend,
            "precision": args.precision,
            "adapter_strategy": args.adapter_strategy,
            "bigwig_dir": str(bigwig_dir),
            "num_bigwigs": len(bigwigs),
            "fasta_path": str(fasta_path),
            "split_source": args.split_source,
            "window_size": args.window_size,
            "stride": args.stride,
            "target_workers": args.target_workers,
            "window_workers": (
                args.window_workers
                if args.window_workers is not None
                else min(args.batch_size, _available_cpu_count())
            ),
            "target_cache_dir": str(target_cache_dir) if target_cache_dir else None,
            "target_cache_dtype": args.target_cache_dtype,
            "backbone_lora": args.backbone_lora,
            "lora_rank": args.lora_rank if args.backbone_lora else None,
            "lora_alpha": args.lora_alpha if args.backbone_lora else None,
            "fp8_lora": args.fp8_lora if args.backbone_lora else None,
            "fp4_lora": args.fp4_lora if args.backbone_lora else None,
            "base_param_dtype": args.base_param_dtype if args.backbone_lora else None,
            "lora_param_dtype": args.lora_param_dtype if args.backbone_lora else None,
            "locon_param_dtype": args.locon_param_dtype if args.backbone_lora else None,
            "activation_dtype": args.activation_dtype if args.backbone_lora else None,
            "base_compute_dtype": args.base_compute_dtype if args.backbone_lora else None,
            "lora_compute_dtype": lora_compute_dtype if args.backbone_lora else None,
        },
        num_devices=args.num_devices,
        eval_splits=tuple(item.strip() for item in args.eval_splits.split(",") if item.strip()),
        progress_interval=args.progress_interval,
        prefetch_batches=args.prefetch_batches,
        profile_host_timing=args.profile_host_timing,
        start_epoch=start_epoch,
        initial_global_step=initial_global_step,
        initial_optimizer_state_path=initial_optimizer_state_path,
    )


if __name__ == "__main__":
    main()
