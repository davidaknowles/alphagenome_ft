"""Find coverage artifacts that are extreme in every track of a head.

Sequencing and mapping artifacts, such as reads piling up on poly-G runs,
appear as bins with very high signal in every cell group. A few such bins can
dominate double-centered correlation metrics. Bins are summarized by the
minimum across tracks, so a strong cell-type-specific peak is not flagged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyBigWig


def binned_minimum(paths: Sequence[Path], chrom: str, length: int, bin_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin minimum and mean across tracks of the mean signal in each bin."""
    n_bins = length // bin_size
    if n_bins == 0:
        return np.zeros(0), np.zeros(0)
    minimum = np.full(n_bins, np.inf)
    total = np.zeros(n_bins)
    for path in paths:
        handle = pyBigWig.open(str(path))
        try:
            values = np.zeros(n_bins)
            if chrom in handle.chroms():
                end = min(n_bins * bin_size, handle.chroms()[chrom] // bin_size * bin_size)
                bases = np.nan_to_num(handle.values(chrom, 0, end, numpy=True))
                values[: end // bin_size] = bases.reshape(-1, bin_size).mean(axis=1)
        finally:
            handle.close()
        minimum = np.minimum(minimum, values)
        total += values
    return minimum, total / len(paths)


def scan_extreme_bins(
    paths: Sequence[Path],
    chrom_sizes: Mapping[str, int],
    *,
    bin_size: int = 128,
    keep_top: int = 5000,
    block_size: int = 131072,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Extreme bins, extreme blocks, and quantiles of the per-bin minimum.

    Returns the ``keep_top`` bins with the largest across-track minimum
    (columns chrom, start, end, minimum, mean), the ``keep_top`` blocks of
    ``block_size`` with the largest summed across-track mean (columns chrom,
    start, end, block_mean_sum), and the genome-wide quantiles at 0.5, 0.9,
    0.99, 0.999, 0.9999 and 0.99999 of the per-bin minimum.
    """
    tops, blocks, minima = [], [], []
    per_block = block_size // bin_size
    for chrom, length in chrom_sizes.items():
        minimum, mean = binned_minimum(paths, chrom, length, bin_size)
        minima.append(minimum)
        order = np.argsort(minimum)[::-1][:keep_top]
        tops.append(pd.DataFrame({"chrom": chrom, "start": order * bin_size, "end": order * bin_size + bin_size, "minimum": minimum[order], "mean": mean[order]}))
        n_blocks = len(mean) // per_block
        if n_blocks:
            sums = mean[: n_blocks * per_block].reshape(n_blocks, per_block).sum(axis=1)
            blocks.append(pd.DataFrame({"chrom": chrom, "start": np.arange(n_blocks) * block_size, "end": np.arange(1, n_blocks + 1) * block_size, "block_mean_sum": sums}))
    quantiles = np.quantile(np.concatenate(minima), [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999])
    top = pd.concat(tops).sort_values("minimum", ascending=False).head(keep_top).reset_index(drop=True)
    block_table = pd.concat(blocks).sort_values("block_mean_sum", ascending=False).reset_index(drop=True)
    return top, block_table, quantiles


def blacklist_regions(top: pd.DataFrame, threshold: float, *, pad: int = 0) -> pd.DataFrame:
    """Merge flagged bins (minimum at or above ``threshold``) into padded regions."""
    flagged = top[top.minimum >= threshold].sort_values(["chrom", "start"])
    rows = []
    for chrom, group in flagged.groupby("chrom", sort=True):
        start, end = None, None
        for bin_start, bin_end in zip(group.start, group.end):
            bin_start, bin_end = max(0, bin_start - pad), bin_end + pad
            if start is not None and bin_start <= end:
                end = max(end, bin_end)
                continue
            if start is not None:
                rows.append((chrom, start, end))
            start, end = bin_start, bin_end
        rows.append((chrom, start, end))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def extreme_blocks(blocks: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Blocks whose summed signal is at least ``factor`` times the median block."""
    threshold = factor * blocks.block_mean_sum.median()
    return blocks[blocks.block_mean_sum >= threshold][["chrom", "start", "end"]].reset_index(drop=True)
