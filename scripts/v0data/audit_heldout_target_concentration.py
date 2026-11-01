#!/usr/bin/env python
"""Audit how concentrated held-out coverage targets are across windows and bins.

For one cached head and split, targets are averaged into 128 bp bins and
double-centered over all (window, bin) rows and cell-group columns, as in the
signed double-centered correlation metric. The script reports each window's
share of the total double-centered sum of squares, the largest single-bin
share, and the windows that dominate. A held-out set whose metric is driven by
a few windows gives unstable validation and test correlations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def binned(array: np.ndarray, bin_size: int = 128) -> np.ndarray:
    """Mean over bins for each window, shape (windows, length / bin_size, groups)."""
    windows, length, groups = array.shape
    out = np.empty((windows, length // bin_size, groups), dtype=np.float64)
    for index in range(windows):
        out[index] = np.asarray(array[index], dtype=np.float64).reshape(length // bin_size, bin_size, groups).mean(axis=1)
    return out


def concentration(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-window and per-bin shares of the double-centered sum of squares."""
    rows = values.reshape(-1, values.shape[-1])
    centered = rows - rows.mean(axis=0, keepdims=True) - rows.mean(axis=1, keepdims=True) + rows.mean()
    squares = (centered**2).sum(axis=1).reshape(values.shape[0], values.shape[1])
    total = squares.sum()
    return squares.sum(axis=1) / total, squares / total, float(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True, help="Target cache datasets directory")
    parser.add_argument("--source", action="append", required=True, help="dataset/source/head, e.g. zemke2023/macaque/zemke2023_atac_macaque")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries, top_rows = [], []
    for value in args.source:
        dataset, source, head = value.split("/")
        root = args.cache_root / dataset / source
        manifest = json.loads((root / "manifest.json").read_text())
        for split in ("valid", "test"):
            intervals = manifest["splits"][split]
            values = binned(np.load(root / split / f"{head}.npy", mmap_mode="r"))
            window_share, bin_share, total = concentration(values)
            order = np.argsort(window_share)[::-1]
            summaries.append({
                "source": f"{dataset}:{source}",
                "head": head,
                "split": split,
                "windows": len(intervals),
                "top1_window_share": float(window_share[order[0]]),
                "top10_window_share": float(window_share[order[:10]].sum()),
                "max_bin_share": float(bin_share.max()),
                "windows_for_half": int(np.searchsorted(np.cumsum(window_share[order]), 0.5) + 1),
            })
            for rank, index in enumerate(order[:5], start=1):
                interval = intervals[index]
                peak_bin = int(np.argmax(bin_share[index]))
                top_rows.append({
                    "source": f"{dataset}:{source}",
                    "split": split,
                    "rank": rank,
                    "chromosome": interval["chromosome"],
                    "start": interval["start"],
                    "end": interval["end"],
                    "window_share": float(window_share[index]),
                    "peak_bin_start": interval["start"] + 128 * peak_bin,
                    "peak_bin_share": float(bin_share[index, peak_bin]),
                })
            print(summaries[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.output.with_suffix(".summary.tsv"), sep="\t", index=False)
    pd.DataFrame(top_rows).to_csv(args.output.with_suffix(".top_windows.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
