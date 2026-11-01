#!/usr/bin/env python
"""Build Zemke 2023 ATAC artifact blacklists and a dataset config that excludes them.

A region is blacklisted if a 128 bp bin's minimum signal across all cell groups
is at least ``--bin-factor`` times the genome-wide 99.99th percentile of that
minimum (padded by ``--pad``), or if a 131,072 bp window's summed signal is at
least ``--block-factor`` times the median window. Windows overlapping a region
are dropped from every split of the matching Zemke 2023 source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from alphagenome_ft.finetune.target_artifacts import blacklist_regions, extreme_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--input-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--bin-factor", type=float, default=10.0)
    parser.add_argument("--block-factor", type=float, default=10.0)
    parser.add_argument("--pad", type=int, default=1000)
    args = parser.parse_args()
    config = json.loads(args.input_config.read_text())
    zemke = next(d for d in config["datasets"] if d["name"] == "zemke2023")
    summary = {}
    for source in zemke["sources"]:
        name = source["name"]
        q9999 = json.loads((args.scan_dir / f"{name}_quantiles.json").read_text())["q9999"]
        bins = pd.read_csv(args.scan_dir / f"{name}_extreme_bins.tsv.gz", sep="\t")
        blocks = pd.read_csv(args.scan_dir / f"{name}_blocks.tsv.gz", sep="\t")
        bin_regions = blacklist_regions(bins, args.bin_factor * q9999, pad=args.pad).assign(reason="extreme_bin")
        block_regions = extreme_blocks(blocks, args.block_factor).assign(reason="extreme_window")
        regions = pd.concat([bin_regions, block_regions]).sort_values(["chrom", "start"])
        bed = args.scan_dir / f"{name}_blacklist.bed"
        regions.to_csv(bed, sep="\t", header=False, index=False)
        source["exclude_intervals_bed"] = str(bed.resolve())
        summary[name] = {"bin_threshold": args.bin_factor * q9999, "bin_regions": len(bin_regions), "window_regions": len(block_regions)}
    config["artifact_blacklist"] = {"bin_factor": args.bin_factor, "block_factor": args.block_factor, "pad": args.pad, "sources": summary}
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
