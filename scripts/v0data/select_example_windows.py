#!/usr/bin/env python
"""Choose example held-out windows by rank of per-window ATAC R.

Reads the ``--score-split`` outputs of ``predict_example_windows.py``. A window is
eligible if at least ``--min-genes`` genes lie fully inside it and reach
``--min-cpm`` in some cell group. For each requested percentile of ATAC R over
all scored windows, the eligible window with the closest R is chosen. The genes
shown for each window are the expressed genes inside it with the largest
across-group standard deviation of observed log10(1 + CPM), so gene choice does
not depend on predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="Stem, e.g. results/v0data/example_tracks/zemke2023_human_test")
    parser.add_argument("--percentile", action="append", required=True, help="NAME=percentile, e.g. high=90")
    parser.add_argument("--min-genes", type=int, default=2)
    parser.add_argument("--min-cpm", type=float, default=10.0)
    parser.add_argument("--genes-shown", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = pd.read_csv(f"{args.scores}_windows.tsv.gz", sep="\t").dropna(subset=["atac_r"])
    genes = pd.read_csv(f"{args.scores}_genes.tsv.gz", sep="\t")
    expressed = genes[genes.inside & (genes.observed_max_cpm >= args.min_cpm)]
    counts = expressed.groupby(["chrom", "start"]).size().rename("n_expressed").reset_index()
    eligible = windows.merge(counts, on=["chrom", "start"])
    eligible = eligible[eligible.n_expressed >= args.min_genes]
    rows = []
    for value in args.percentile:
        name, _, q = value.partition("=")
        target = float(np.percentile(windows.atac_r, float(q)))
        pick = eligible.iloc[(eligible.atac_r - target).abs().argmin()]
        shown = expressed[(expressed.chrom == pick.chrom) & (expressed.start == pick.start)].nlargest(args.genes_shown, "observed_log_sd")
        rank = float((windows.atac_r < pick.atac_r).mean() * 100)
        rows.append({"name": name, "percentile": float(q), "target_r": target, "chrom": pick.chrom, "start": int(pick.start), "atac_r": float(pick.atac_r), "rank_percentile": rank, "genes": ",".join(shown.gene), "gene_r": ",".join(f"{r:.3f}" for r in shown.rna_r)})
    table = pd.DataFrame(rows)
    table.to_csv(args.output, sep="\t", index=False)
    print(f"scored windows {len(windows)}, eligible {len(eligible)}, median R {windows.atac_r.median():.3f}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
