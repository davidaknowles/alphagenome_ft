#!/usr/bin/env python
"""Build a HumanBrainDev target cache for selected chromosome splits only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import (
    BigWigDataModule,
    WindowedTargetCache,
    load_targets_config,
    prepare_head_specs,
    register_predefined_heads,
    validate_head_specs,
)
from scripts.run_humanbraindev_finetune import (
    DEFAULT_FASTA,
    build_targets_config,
    discover_bigwigs,
    make_chromosome_split_intervals,
    parse_chrom_set,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bigwig-dir",
        type=Path,
        default=Path("/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs"),
    )
    parser.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--head-id", default="humanbraindev_atac")
    parser.add_argument("--window-size", type=int, default=1_048_576)
    parser.add_argument("--stride", type=int, default=1_048_576)
    parser.add_argument("--valid-chroms", default="chr8")
    parser.add_argument("--test-chroms", default="chr9")
    parser.add_argument("--exclude-chroms", default="chrM,chrY")
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--target-cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--target-cache-workers", type=int, default=32)
    parser.add_argument("--overwrite-target-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        limit_train=None,
        limit_valid=args.limit_valid,
        limit_test=args.limit_test,
    )
    selected = {split.strip() for split in args.splits.split(",") if split.strip()}
    intervals = {split: values for split, values in intervals.items() if split in selected}
    if not intervals:
        raise ValueError(f"No intervals for selected split(s): {sorted(selected)}")
    for split, split_intervals in intervals.items():
        print(f"{split}: {len(split_intervals)} interval(s)")

    filtered = BigWigDataModule._filter_intervals_by_bigwig_chromosomes(intervals, head_specs)
    WindowedTargetCache.build(
        args.target_cache_dir.expanduser().resolve(),
        intervals=filtered,
        head_specs=head_specs,
        dtype=args.target_cache_dtype,
        workers=args.target_cache_workers,
        overwrite=args.overwrite_target_cache,
    )


if __name__ == "__main__":
    main()
