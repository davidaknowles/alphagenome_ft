#!/usr/bin/env python3
"""Audit expanded direct-gene assignment over fixed genomic windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome.data import genome

from alphagenome_ft.finetune import load_targets_config
from alphagenome_ft.finetune.config import prepare_head_specs
from alphagenome_ft.finetune.data import GeneExpressionSupervision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-config", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--fasta-index", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=131_072)
    parser.add_argument("--json-output", required=True, type=Path)
    args = parser.parse_args()
    config = load_targets_config(args.targets_config)
    specs = [spec for spec in prepare_head_specs(config) if spec.head_id == args.head]
    if len(specs) != 1 or specs[0].gene_supervision_path is None:
        raise ValueError(f'Expected one gene-supervised head named "{args.head}".')
    chromosome_sizes = {
        fields[0]: int(fields[1])
        for line in args.fasta_index.read_text().splitlines()
        if len(fields := line.split("\t")) >= 2
    }
    windows = {
        "all": [
            genome.Interval(chromosome, start, start + args.window_size)
            for chromosome, size in chromosome_sizes.items()
            if "_" not in chromosome and chromosome not in {"chrM", "chrY"}
            for start in range(0, size - args.window_size + 1, args.window_size)
        ]
    }
    supervision = GeneExpressionSupervision(specs[0].gene_supervision_path, specs[0])
    supervision.configure_windows(windows)
    result = {
        "targets_config": str(args.targets_config),
        "head": args.head,
        "window_size": args.window_size,
        **supervision.assignment_summary(),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
