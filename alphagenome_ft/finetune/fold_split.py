"""Train/valid/test window splits aligned with AlphaGenome's pretraining folds.

AlphaGenome's per-fold checkpoints hold out two of eight Borzoi genome folds.
Fine-tuning from such a checkpoint and evaluating only on windows inside its
held-out folds gives test sequence the backbone never saw as a training target.
Windows are labeled by overlap with fold regions, either directly (human and
mouse) or through liftover of fixed-size bins to a species with fold regions.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

SPLITS = ("train", "valid", "test")


def alphagenome_fold_regions(model_version: str, organism: str) -> pd.DataFrame:
    """Regions with columns chrom, start, end, split for an AlphaGenome checkpoint."""
    from alphagenome.data import fold_intervals
    from alphagenome.models import dna_client

    version = dna_client.ModelVersion[model_version.upper()]
    organism_enum = dna_client.Organism[organism.upper()]
    subsets = {"train": fold_intervals.Subset.TRAIN, "valid": fold_intervals.Subset.VALID, "test": fold_intervals.Subset.TEST}
    frames = []
    for split, subset in subsets.items():
        regions = fold_intervals.get_fold_intervals(version, organism_enum, subset)
        frames.append(regions.rename(columns={"chromosome": "chrom"})[["chrom", "start", "end"]].assign(split=split))
    return pd.concat(frames, ignore_index=True)


def tile_windows(
    chrom_sizes: Mapping[str, int],
    *,
    window_size: int,
    stride: int,
    include_chroms: set[str] | None = None,
    exclude_chroms: set[str] = frozenset(),
) -> pd.DataFrame:
    """Fixed windows with the same chromosome rules as the chromosome-split loader."""
    rows = []
    for chrom, size in chrom_sizes.items():
        if chrom in exclude_chroms or size < window_size:
            continue
        if include_chroms is not None and chrom not in include_chroms:
            continue
        if include_chroms is None and ("_" in chrom or chrom.startswith("chrUn")):
            continue
        rows.extend((chrom, start, start + window_size) for start in range(0, size - window_size + 1, stride))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def _merge(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    merged_starts, merged_ends = [starts[0]], [ends[0]]
    for start, end in zip(starts[1:], ends[1:]):
        if start <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], end)
        else:
            merged_starts.append(start)
            merged_ends.append(end)
    return np.asarray(merged_starts), np.asarray(merged_ends)


def _overlap_bp(starts: np.ndarray, ends: np.ndarray, region_starts: np.ndarray, region_ends: np.ndarray) -> np.ndarray:
    """Base pairs of each query interval covered by sorted, disjoint regions."""
    cumulative = np.concatenate([[0], np.cumsum(region_ends - region_starts)])

    def covered_before(position: np.ndarray) -> np.ndarray:
        # Regions 0..k-1 start before position; only region k-1 can extend past it.
        k = np.searchsorted(region_starts, position, side="left")
        overhang = np.where(k > 0, np.clip(region_ends[np.maximum(k - 1, 0)] - position, 0, None), 0)
        return cumulative[k] - overhang

    return covered_before(ends) - covered_before(starts)


def split_overlap(intervals: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Base pairs of each interval covered by each split's regions."""
    out = pd.DataFrame(0, index=intervals.index, columns=list(SPLITS), dtype=np.int64)
    for split in SPLITS:
        for chrom, split_regions in regions[regions.split == split].groupby("chrom"):
            mask = (intervals.chrom == chrom).to_numpy()
            if not mask.any():
                continue
            region_starts, region_ends = _merge(split_regions.start.to_numpy(), split_regions.end.to_numpy())
            out.loc[mask, split] = _overlap_bp(
                intervals.start.to_numpy()[mask], intervals.end.to_numpy()[mask], region_starts, region_ends
            )
    return out


def label_from_overlap(overlap: pd.DataFrame, *, total: np.ndarray, min_heldout: float, max_other: float) -> pd.Series:
    """Assign each row one split, or None when it mixes splits.

    A row touching no valid or test region is ``train``. Otherwise it is
    ``valid`` or ``test`` if that split covers at least ``min_heldout`` of
    ``total`` and every other split covers at most ``max_other`` of the
    labeled amount; remaining rows are dropped.
    """
    labeled = overlap.sum(axis=1).to_numpy()
    labels = np.full(len(overlap), None, dtype=object)
    heldout = overlap[["valid", "test"]].sum(axis=1).to_numpy()
    labels[heldout == 0] = "train"
    for split in ("valid", "test"):
        own = overlap[split].to_numpy()
        other = labeled - own
        keep = (heldout > 0) & (own >= min_heldout * total) & (other <= max_other * np.maximum(labeled, 1))
        labels[keep] = split
    return pd.Series(labels, index=overlap.index)


def label_windows_direct(windows: pd.DataFrame, regions: pd.DataFrame, *, min_heldout: float = 0.5, max_other: float = 0.0) -> pd.Series:
    """Label windows on the same assembly as the fold regions."""
    overlap = split_overlap(windows, regions)
    return label_from_overlap(overlap, total=(windows.end - windows.start).to_numpy(), min_heldout=min_heldout, max_other=max_other)


def liftover_bins(
    windows: pd.DataFrame,
    chains: list[Path],
    *,
    bin_size: int = 1000,
    min_match: list[float],
    rename: Mapping[str, str] | None = None,
    liftover: str = "liftOver",
) -> pd.DataFrame:
    """Lift fixed bins of each window through one or more chains.

    Returns columns window, chrom, start, end in the final assembly, one row
    per bin that maps through every chain.
    """
    bins = []
    for index, row in windows.iterrows():
        starts = np.arange(row.start, row.end, bin_size)
        bins.append(pd.DataFrame({"chrom": row.chrom, "start": starts, "end": np.minimum(starts + bin_size, row.end), "window": index}))
    current = pd.concat(bins, ignore_index=True)
    if rename is not None:
        current["chrom"] = current.chrom.map(rename)
        current = current.dropna(subset=["chrom"])
    current["name"] = current.window.astype(str) + ":" + current.start.astype(str)
    with tempfile.TemporaryDirectory() as tmp:
        for step, (chain, match) in enumerate(zip(chains, min_match, strict=True)):
            source = Path(tmp) / f"in{step}.bed"
            target = Path(tmp) / f"out{step}.bed"
            current[["chrom", "start", "end", "name"]].to_csv(source, sep="\t", header=False, index=False)
            subprocess.run([liftover, f"-minMatch={match}", str(source), str(chain), str(target), str(Path(tmp) / f"unmapped{step}.bed")], check=True, capture_output=True)
            current = pd.read_csv(target, sep="\t", header=None, names=["chrom", "start", "end", "name"])
    current["window"] = current.name.str.split(":").str[0].astype(int)
    return current[["window", "chrom", "start", "end"]]


def label_windows_lifted(
    windows: pd.DataFrame,
    lifted: pd.DataFrame,
    regions: pd.DataFrame,
    *,
    bin_size: int = 1000,
    min_heldout: float = 0.2,
    max_other: float = 0.05,
) -> pd.Series:
    """Label windows from the fold regions overlapped by their lifted bins.

    Each lifted bin takes the split covering its midpoint; ``min_heldout`` is a
    fraction of the window's bins and ``max_other`` a fraction of labeled bins.
    """
    midpoints = lifted.assign(start=(lifted.start + lifted.end) // 2).assign(end=lambda d: d.start + 1)
    per_bin = split_overlap(midpoints, regions).clip(upper=1)
    counts = per_bin.groupby(lifted.window.to_numpy()).sum().reindex(windows.index, fill_value=0)
    bins_per_window = np.ceil((windows.end - windows.start).to_numpy() / bin_size)
    return label_from_overlap(counts, total=bins_per_window, min_heldout=min_heldout, max_other=max_other)
