#!/usr/bin/env python
"""Scan one Zemke 2023 ATAC source for coverage artifacts shared by all cell groups.

Writes the most extreme 128 bp bins (by minimum across the 20 tracks), all
131,072 bp blocks ranked by summed signal, and genome-wide quantiles, with the
G/C homopolymer and soft-masked fractions of each extreme bin's sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyfaidx

from alphagenome_ft.finetune.target_artifacts import scan_extreme_bins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Zemke 2023 source name, e.g. macaque")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.dataset_config.read_text())
    source = next(s for d in config["datasets"] if d["name"] == "zemke2023" for s in d["sources"] if s["name"] == args.source)
    targets = json.loads(Path(source["targets_config"]).read_text())
    head = next(h for h in targets["heads"] if "atac" in h["id"])
    paths = [Path(t["path"]) for t in head["targets"]]
    include = [c for c in source.get("include_chroms", "").split(",") if c]
    sizes = {line.split()[0]: int(line.split()[1]) for line in open(f"{source['fasta']}.fai")}
    chrom_sizes = {c: sizes[c] for c in include} if include else sizes
    top, blocks, quantiles = scan_extreme_bins(paths, chrom_sizes)
    fasta = pyfaidx.Fasta(source["fasta"])
    sequences = [str(fasta[row.chrom][int(row.start):int(row.end)]) for row in top.head(500).itertuples()]
    top.loc[: len(sequences) - 1, "softmasked"] = [sum(c.islower() for c in s) / len(s) for s in sequences]
    top.loc[: len(sequences) - 1, "g_or_c_run"] = [int("G" * 30 in s.upper() or "C" * 30 in s.upper()) for s in sequences]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.output_dir / f"{args.source}_extreme_bins.tsv.gz", sep="\t", index=False)
    blocks.to_csv(args.output_dir / f"{args.source}_blocks.tsv.gz", sep="\t", index=False)
    labels = ["q50", "q90", "q99", "q999", "q9999", "q99999"]
    (args.output_dir / f"{args.source}_quantiles.json").write_text(json.dumps(dict(zip(labels, map(float, quantiles))), indent=2))
    print(args.source, dict(zip(labels, np.round(quantiles, 2))), top.head(5).to_dict("records"))


if __name__ == "__main__":
    main()
