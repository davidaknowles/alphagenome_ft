"""Data utilities for fine-tuning."""

from __future__ import annotations

import contextlib
import concurrent.futures
import gzip
import heapq
import json
import os
import resource
import threading
from collections import defaultdict
from pathlib import Path
from bisect import bisect_left
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyBigWig
from alphagenome.data import genome
from alphagenome_research.io import fasta as fasta_lib
from alphagenome_research.model import one_hot_encoder


from alphagenome_ft.finetune.config import HeadSpec

WINDOW_SIZE = 1_048_576  # 1 Mbp windows

_DEFAULT_INTERVALS = {
    "HOMO_SAPIENS": (
        "https://github.com/calico/borzoi/raw/"
        "5c9358222b5026abb733ed5fb84f3f6c77239b37/data/sequences_human.bed.gz"
    ),
    "MUS_MUSCULUS": (
        "https://github.com/calico/borzoi/raw/"
        "5c9358222b5026abb733ed5fb84f3f6c77239b37/data/sequences_mouse.bed.gz"
    ),
}

_ORGANISM_ALIASES = {
    "human": "HOMO_SAPIENS",
    "homo_sapiens": "HOMO_SAPIENS",
    "homo-sapiens": "HOMO_SAPIENS",
    "homo sapiens": "HOMO_SAPIENS",
    "hg38": "HOMO_SAPIENS",
    "mouse": "MUS_MUSCULUS",
    "mus_musculus": "MUS_MUSCULUS",
    "mus-musculus": "MUS_MUSCULUS",
    "mus musculus": "MUS_MUSCULUS",
    "mm10": "MUS_MUSCULUS",
}


class GeneExpressionSupervision:
    """Window-indexed exon geometry and gene-by-track expression targets."""

    def __init__(self, path: Path, spec: HeadSpec) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.gene_ids = tuple(str(value) for value in data["gene_ids"])
            self.chromosomes = np.asarray(data["chromosomes"]).astype(str)
            self.starts = np.asarray(data["starts"], dtype=np.int64)
            self.ends = np.asarray(data["ends"], dtype=np.int64)
            self.strands = np.asarray(data["strands"]).astype(str)
            self.exon_offsets = np.asarray(data["exon_offsets"], dtype=np.int64)
            self.exon_starts = np.asarray(data["exon_starts"], dtype=np.int64)
            self.exon_ends = np.asarray(data["exon_ends"], dtype=np.int64)
            self.groups = tuple(str(value) for value in data["groups"])
            self.cpm = np.asarray(data["cpm"], dtype=np.float32)
            self.group_valid = (
                np.asarray(data["group_valid"], dtype=bool)
                if "group_valid" in data
                else np.ones((len(self.groups),), dtype=bool)
            )
        gene_count = len(self.gene_ids)
        if self.cpm.shape != (len(self.groups), gene_count):
            raise ValueError(f"Invalid gene CPM shape {self.cpm.shape} in {path}.")
        if self.group_valid.shape != (len(self.groups),):
            raise ValueError(f"Invalid group-valid shape {self.group_valid.shape} in {path}.")
        if not np.any(self.group_valid):
            raise ValueError(f"Gene supervision has no valid groups in {path}.")
        self.window_assignment = spec.gene_window_assignment
        self._assigned_indices: dict[tuple[str, int, int], np.ndarray] = {}
        self._assigned_scales: dict[tuple[str, int, int], np.ndarray] = {}
        track_strands = tuple(track.strand for track in spec.tracks)
        paired_strands = tuple(strand for _ in self.groups for strand in ("+", "-"))
        if len(spec.tracks) == len(self.groups) and set(track_strands) <= {".", ""}:
            self.stranded = False
        elif len(spec.tracks) == 2 * len(self.groups) and track_strands == paired_strands:
            self.stranded = True
        else:
            raise ValueError(
                f"Head {spec.head_id} must contain one unstranded track or an interleaved "
                f"+/- track pair for each of its {len(self.groups)} gene groups."
            )
        self._by_chromosome: dict[str, np.ndarray] = {}
        for chromosome in np.unique(self.chromosomes):
            indices = np.flatnonzero(self.chromosomes == chromosome)
            self._by_chromosome[str(chromosome)] = indices[np.argsort(self.starts[indices])]

    def contained_indices(self, window: genome.Interval) -> np.ndarray:
        indices = self._by_chromosome.get(window.chromosome)
        if indices is None:
            return np.empty((0,), dtype=np.int64)
        starts = self.starts[indices]
        upper = np.searchsorted(starts, window.end, side="right")
        candidates = indices[:upper]
        return candidates[
            (self.starts[candidates] >= window.start) & (self.ends[candidates] <= window.end)
        ]

    @staticmethod
    def _window_key(window: genome.Interval) -> tuple[str, int, int]:
        return window.chromosome, int(window.start), int(window.end)

    def configure_windows(self, intervals: Mapping[str, Sequence[genome.Interval]]) -> None:
        """Assign each gene once for partial-exon supervision."""
        if self.window_assignment == "full_span":
            return
        windows_by_chromosome: dict[str, list[genome.Interval]] = defaultdict(list)
        for split_windows in intervals.values():
            for window in split_windows:
                windows_by_chromosome[window.chromosome].append(window)
        assigned: dict[tuple[str, int, int], list[tuple[int, float]]] = defaultdict(list)
        for chromosome, windows in windows_by_chromosome.items():
            windows.sort(key=lambda window: (window.start, window.end))
            if any(left.end > right.start for left, right in zip(windows, windows[1:])):
                raise ValueError(
                    "max_exon_overlap_scaled requires non-overlapping sequence windows."
                )
            window_starts = np.asarray([window.start for window in windows], dtype=np.int64)
            window_ends = np.asarray([window.end for window in windows], dtype=np.int64)
            for gene_idx in self._by_chromosome.get(chromosome, ()):
                exon_begin = self.exon_offsets[gene_idx]
                exon_end = self.exon_offsets[gene_idx + 1]
                scores: dict[int, int] = defaultdict(int)
                total_exon_bases = 0
                for start, end in zip(
                    self.exon_starts[exon_begin:exon_end],
                    self.exon_ends[exon_begin:exon_end],
                    strict=True,
                ):
                    total_exon_bases += int(end - start)
                    first_window = int(np.searchsorted(window_ends, start, side="right"))
                    for window_idx in range(first_window, len(windows)):
                        if window_starts[window_idx] >= end:
                            break
                        overlap = min(int(end), int(window_ends[window_idx])) - max(
                            int(start), int(window_starts[window_idx])
                        )
                        if overlap > 0:
                            scores[window_idx] += overlap
                if not scores or total_exon_bases <= 0:
                    continue
                best_window_idx, observed_exon_bases = min(
                    scores.items(), key=lambda item: (-item[1], windows[item[0]].start)
                )
                assigned[self._window_key(windows[best_window_idx])].append(
                    (int(gene_idx), total_exon_bases / observed_exon_bases)
                )
        self._assigned_indices = {
            key: np.asarray([gene_idx for gene_idx, _ in values], dtype=np.int64)
            for key, values in assigned.items()
        }
        self._assigned_scales = {
            key: np.asarray([scale for _, scale in values], dtype=np.float32)
            for key, values in assigned.items()
        }

    def indices_for_window(self, window: genome.Interval) -> np.ndarray:
        if self.window_assignment == "full_span":
            return self.contained_indices(window)
        return self._assigned_indices.get(
            self._window_key(window), np.empty((0,), dtype=np.int64)
        )

    def assignment_summary(self) -> dict[str, object]:
        """Summarize expanded gene support and exon-length extrapolation scales."""
        if self.window_assignment == "full_span":
            raise ValueError("Assignment summary requires an expanded window assignment.")
        assigned_indices = (
            np.concatenate(tuple(self._assigned_indices.values()))
            if self._assigned_indices
            else np.empty((0,), dtype=np.int64)
        )
        scales = (
            np.concatenate(tuple(self._assigned_scales.values()))
            if self._assigned_scales
            else np.empty((0,), dtype=np.float32)
        )
        if len(np.unique(assigned_indices)) != len(assigned_indices):
            raise RuntimeError("Expanded gene assignment contains duplicate genes.")

        def summarize(selected_scales: np.ndarray) -> dict[str, float | int]:
            quantiles = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
            return {
                "assigned_genes": int(len(selected_scales)),
                "scale_quantiles": (
                    {
                        str(quantile): float(np.quantile(selected_scales, quantile))
                        for quantile in quantiles
                    }
                    if len(selected_scales)
                    else {}
                ),
            }

        by_chromosome = {}
        for chromosome in sorted(set(self.chromosomes)):
            chromosome_mask = self.chromosomes[assigned_indices] == chromosome
            by_chromosome[chromosome] = {
                "available_genes": int(np.sum(self.chromosomes == chromosome)),
                **summarize(scales[chromosome_mask]),
            }
        return {
            "available_genes": len(self.gene_ids),
            **summarize(scales),
            "chromosomes": by_chromosome,
        }

    def max_genes(self, intervals: Mapping[str, Sequence[genome.Interval]]) -> int:
        return max(
            (
                len(self.indices_for_window(window))
                for windows in intervals.values()
                for window in windows
            ),
            default=0,
        )

    def arrays_for_window(
        self,
        window: genome.Interval,
        *,
        sequence_length: int,
        max_genes: int,
    ) -> dict[str, np.ndarray]:
        if sequence_length % 128:
            raise ValueError(
                "Gene expression supervision requires sequence length divisible by 128."
            )
        indices = self.indices_for_window(window)
        scales = (
            np.ones(len(indices), dtype=np.float32)
            if self.window_assignment == "full_span"
            else self._assigned_scales.get(
                self._window_key(window), np.empty((0,), dtype=np.float32)
            )
        )
        if len(indices) > max_genes:
            raise ValueError(
                f"Window contains {len(indices)} genes, exceeding configured {max_genes}."
            )
        weights = np.zeros((sequence_length // 128, max_genes), dtype=np.float32)
        targets = np.zeros((max_genes, len(self.groups)), dtype=np.float32)
        strands = np.zeros((max_genes,), dtype=np.int8)
        has_missing_groups = not np.all(self.group_valid)
        valid = np.zeros(
            (max_genes, len(self.groups)) if has_missing_groups else (max_genes,),
            dtype=bool,
        )
        for output_idx, (gene_idx, scale) in enumerate(zip(indices, scales, strict=True)):
            exon_begin = self.exon_offsets[gene_idx]
            exon_end = self.exon_offsets[gene_idx + 1]
            for start, end in zip(
                self.exon_starts[exon_begin:exon_end],
                self.exon_ends[exon_begin:exon_end],
                strict=True,
            ):
                local_start = max(0, int(start) - window.start)
                local_end = min(sequence_length, int(end) - window.start)
                if local_end <= local_start:
                    continue
                first_bin = local_start // 128
                last_bin = (local_end - 1) // 128
                for bin_idx in range(first_bin, last_bin + 1):
                    overlap = min(local_end, (bin_idx + 1) * 128) - max(local_start, bin_idx * 128)
                    weights[bin_idx, output_idx] += overlap / 128.0
            weights[:, output_idx] *= scale
            targets[output_idx] = self.cpm[:, gene_idx]
            strands[output_idx] = 0 if self.strands[gene_idx] == "+" else 1
            valid[output_idx] = self.group_valid if has_missing_groups else True
        return {"weights": weights, "targets": targets, "strands": strands, "valid": valid}


def _balance_gene_window_order(
    order: np.ndarray,
    gene_counts: np.ndarray,
    *,
    batch_size: int,
    drop_last: bool,
) -> np.ndarray:
    """Distribute gene-bearing windows across fixed-size batches without resampling."""
    order = np.asarray(order, dtype=np.int64)
    gene_counts = np.asarray(gene_counts, dtype=np.int64)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if gene_counts.ndim != 1 or np.any(gene_counts < 0):
        raise ValueError("gene_counts must be a non-negative vector.")
    if np.any(order < 0) or np.any(order >= len(gene_counts)):
        raise ValueError("order contains an out-of-range window index.")
    usable = len(order) // batch_size * batch_size if drop_last else len(order)
    selected = order[:usable]
    if len(selected) == 0:
        return selected
    num_batches = (
        len(selected) // batch_size
        if drop_last
        else (len(selected) + batch_size - 1) // batch_size
    )
    capacities = np.full(num_batches, batch_size, dtype=np.int64)
    if not drop_last and len(selected) % batch_size:
        capacities[-1] = len(selected) % batch_size

    positive = selected[gene_counts[selected] > 0]
    negative = selected[gene_counts[selected] == 0]
    positive = positive[np.argsort(-gene_counts[positive], kind="stable")]
    batches: list[list[int]] = [[] for _ in range(num_batches)]
    available_batches = [(0, batch_index) for batch_index in range(num_batches)]
    heapq.heapify(available_batches)
    for index in positive:
        current_gene_count, destination = heapq.heappop(available_batches)
        batches[destination].append(int(index))
        if len(batches[destination]) < capacities[destination]:
            heapq.heappush(
                available_batches,
                (current_gene_count + int(gene_counts[index]), destination),
            )

    negative_index = 0
    for batch_index, batch in enumerate(batches):
        remaining = int(capacities[batch_index] - len(batch))
        batch.extend(int(index) for index in negative[negative_index : negative_index + remaining])
        negative_index += remaining
    if negative_index != len(negative):
        raise RuntimeError("Failed to assign every selected gene-free window.")
    balanced = np.asarray([index for batch in batches for index in batch], dtype=np.int64)
    if len(balanced) != len(selected) or not np.array_equal(
        np.sort(balanced), np.sort(selected)
    ):
        raise RuntimeError("Balanced window order did not preserve selected windows exactly once.")
    return balanced


def _repeat_gene_window_order(
    order: np.ndarray,
    gene_counts: np.ndarray,
    *,
    additional_repeats: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Append repeated gene-bearing windows while retaining the complete base order."""
    order = np.asarray(order, dtype=np.int64)
    gene_counts = np.asarray(gene_counts, dtype=np.int64)
    if additional_repeats < 0:
        raise ValueError("additional_repeats must be non-negative.")
    if gene_counts.ndim != 1 or np.any(gene_counts < 0):
        raise ValueError("gene_counts must be a non-negative vector.")
    if np.any(order < 0) or np.any(order >= len(gene_counts)):
        raise ValueError("order contains an out-of-range window index.")
    if additional_repeats == 0:
        return order.copy()
    gene_order = order[gene_counts[order] > 0]
    repeated = np.tile(gene_order, additional_repeats)
    combined = np.concatenate((order, repeated))
    if rng is not None:
        rng.shuffle(combined)
    return combined


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


def _interval_record(interval: genome.Interval) -> dict[str, int | str | bool]:
    return {
        "chromosome": interval.chromosome,
        "start": int(interval.start),
        "end": int(interval.end),
        "negative_strand": bool(interval.negative_strand),
    }


def _track_record(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.expanduser().resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _json_dump(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _json_load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _normalize_cache_dtype(dtype: str | np.dtype) -> np.dtype:
    normalized = np.dtype(dtype)
    if normalized not in {np.dtype("float16"), np.dtype("float32")}:
        raise ValueError(f"target_cache_dtype must be float16 or float32, got {normalized.name}.")
    return normalized


# https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes
# https://hgdownload.cse.ucsc.edu/goldenpath/mm10/bigZips/mm10.chrom.sizes
_CHROMSIZES = {
    "HOMO_SAPIENS": {
        "chr1": 248_956_422,
        "chr2": 242_193_529,
        "chr3": 198_295_559,
        "chr4": 190_214_555,
        "chr5": 181_538_259,
        "chr6": 170_805_979,
        "chr7": 159_345_973,
        "chr8": 145_138_636,
        "chr9": 138_394_717,
        "chr10": 133_797_422,
        "chr11": 135_086_622,
        "chr12": 133_275_309,
        "chr13": 114_364_328,
        "chr14": 107_043_718,
        "chr15": 101_991_189,
        "chr16": 90_338_345,
        "chr17": 83_257_441,
        "chr18": 80_373_285,
        "chr19": 58_617_616,
        "chr20": 64_444_167,
        "chr21": 46_709_983,
        "chr22": 50_818_468,
        "chrX": 156_040_895,
        "chrY": 57_227_415,
    },
    "MUS_MUSCULUS": {
        "chr1": 195_471_971,
        "chr2": 182_113_224,
        "chr3": 160_039_680,
        "chr4": 156_508_116,
        "chr5": 151_834_684,
        "chr6": 149_736_546,
        "chr7": 145_441_459,
        "chr8": 129_401_213,
        "chr9": 124_595_110,
        "chr10": 130_694_993,
        "chr11": 122_082_543,
        "chr12": 120_129_022,
        "chr13": 120_421_639,
        "chr14": 124_902_244,
        "chr15": 104_043_685,
        "chr16": 98_207_768,
        "chr17": 94_987_271,
        "chr18": 90_702_639,
        "chr19": 61_431_566,
        "chrX": 171_031_299,
        "chrY": 91_744_698,
    },
}

FOLD_MAPPING = {
    "0": {
        "train": ["fold2", "fold3", "fold4", "fold5", "fold6", "fold7"],
        "valid": ["fold1"],
        "test": ["fold0"],
    },
    "1": {
        "train": ["fold0", "fold3", "fold4", "fold5", "fold6", "fold7"],
        "valid": ["fold2"],
        "test": ["fold1"],
    },
    "2": {
        "train": ["fold0", "fold1", "fold4", "fold5", "fold6", "fold7"],
        "valid": ["fold3"],
        "test": ["fold2"],
    },
    "3": {
        "train": ["fold0", "fold1", "fold2", "fold5", "fold6", "fold7"],
        "valid": ["fold4"],
        "test": ["fold3"],
    },
}


def build_split_lookup(fold_key: str) -> dict[str, str]:
    split_lookup: dict[str, str] = {}
    mapping = FOLD_MAPPING[fold_key]
    for split_name, fold_list in mapping.items():
        for fold_label in fold_list:
            split_lookup[fold_label] = split_name
    return split_lookup


def expand_interval(
    start: int,
    end: int,
    *,
    window_size: int,
    chrom_size: int | None,
) -> tuple[int, int]:
    length = end - start
    if length <= 0:
        raise ValueError(f"Invalid interval with non-positive length: start={start}, end={end}")

    midpoint = start + length // 2
    half_window = window_size // 2
    new_start = midpoint - half_window
    new_end = new_start + window_size

    if new_start < 0:
        new_start = 0
        new_end = window_size

    if chrom_size is not None and new_end > chrom_size:
        new_end = chrom_size
        new_start = new_end - window_size
        if new_start < 0:
            raise ValueError(
                "Chromosome length is shorter than the requested window size; cannot place interval."
            )

    return new_start, new_end


def _normalize_fold_label(fold_label: object) -> str:
    label = str(fold_label).strip()
    if label.startswith("fold"):
        return label
    return f"fold{label}"


def _normalize_organism(organism: str) -> str:
    raw = str(organism).strip()
    if raw in _DEFAULT_INTERVALS:
        return raw
    key = raw.lower()
    return _ORGANISM_ALIASES.get(key, raw)


def _has_training_overlap(
    chrom: str,
    start: int,
    end: int,
    training_intervals: dict[str, tuple[Sequence[tuple[int, int]], Sequence[int]]],
) -> bool:
    record = training_intervals.get(chrom)
    if not record:
        return False

    intervals, starts = record
    idx = bisect_left(starts, start)

    def overlaps(interval: tuple[int, int]) -> bool:
        return interval[0] < end and interval[1] > start

    if idx < len(intervals) and overlaps(intervals[idx]):
        return True
    if idx > 0 and overlaps(intervals[idx - 1]):
        return True
    return False


def get_fold_split(
    fold: str | int,
    window_size: int = WINDOW_SIZE,
    organism: str = "HOMO_SAPIENS",
    bed_path: str | None = None,
) -> pd.DataFrame:
    """Create train/valid/test windows for a Borzoi-style fold split.

    Args:
        fold: Fold identifier (``0``-``3``) used with ``FOLD_MAPPING``.
        window_size: Final window length centered on each source interval.
        organism: Organism key or alias (for example ``HOMO_SAPIENS`` or ``hg38``).
        bed_path: Optional BED/BED.GZ path containing ``chrom start end fold``.
            If omitted, uses a built-in default BED for the selected organism.

    Returns:
        DataFrame with columns ``chromosome``, ``start``, ``end``, ``split``.
        Validation/test windows that overlap any training window are removed.

    Raises:
        ValueError: If fold/organism labels are invalid, BED content is empty,
            or fold labels in the BED do not match the selected mapping.
    """
    organism = _normalize_organism(organism)

    fold_key = str(fold)
    if fold_key not in FOLD_MAPPING:
        valid = ", ".join(sorted(FOLD_MAPPING))
        raise ValueError(f"Invalid fold '{fold}'. Valid folds: {valid}")

    if bed_path is None:
        if organism not in _DEFAULT_INTERVALS:
            valid = ", ".join(sorted(_DEFAULT_INTERVALS))
            raise ValueError(f"Unknown organism '{organism}'. Valid values: {valid}")
        bed_path = _DEFAULT_INTERVALS[organism]

    regions = pd.read_csv(
        bed_path,
        sep="\t",
        names=["chromosome", "start", "end", "fold"],
        comment="#",
    )
    if regions.empty:
        raise ValueError("Input BED did not contain any intervals.")

    split_lookup = build_split_lookup(fold_key)

    windows: list[tuple[str, int, int, str]] = []
    for _, row in regions.iterrows():
        chrom = str(row["chromosome"])
        start = int(row["start"])
        end = int(row["end"])
        fold_label = _normalize_fold_label(row["fold"])

        split = split_lookup.get(fold_label)
        if split is None:
            raise ValueError(
                f"Input fold label '{fold_label}' is not recognized for selected fold {fold_key}."
            )

        chrom_size = _CHROMSIZES[organism].get(chrom)
        new_start, new_end = expand_interval(
            start,
            end,
            window_size=window_size,
            chrom_size=chrom_size,
        )
        windows.append((chrom, new_start, new_end, split))

    training_by_chrom: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, end, split in windows:
        if split == "train":
            training_by_chrom.setdefault(chrom, []).append((start, end))

    training_lookup: dict[str, tuple[Sequence[tuple[int, int]], Sequence[int]]] = {}
    for chrom, intervals in training_by_chrom.items():
        intervals.sort(key=lambda iv: iv[0])
        starts = [iv[0] for iv in intervals]
        training_lookup[chrom] = (intervals, starts)

    filtered: list[tuple[str, int, int, str]] = []
    for chrom, start, end, split in windows:
        if split in {"valid", "test"} and _has_training_overlap(chrom, start, end, training_lookup):
            continue
        filtered.append((chrom, start, end, split))

    num_filtered = len(windows) - len(filtered)
    if num_filtered > 0:
        print(
            f"Filtered out {num_filtered} intervals from valid/test sets due to training overlap."
        )

    return pd.DataFrame(filtered, columns=["chromosome", "start", "end", "split"])


def build_interval(
    *, chromosome: str, start: int, end: int, window_size: int | None = None
) -> genome.Interval:
    if start >= end:
        raise ValueError(f"Invalid interval ({chromosome}, {start}, {end}).")
    if window_size is not None:
        center = (start + end) // 2
        half = window_size // 2
        start = max(0, center - half)
        end = start + window_size
    return genome.Interval(start=start, end=end, chromosome=chromosome)


def _maybe_limit(intervals: list[genome.Interval], limit: int | None) -> None:
    if limit is not None and len(intervals) > limit:
        del intervals[limit:]


def _finalize_splits(
    splits: dict[str, list[genome.Interval]],
    *,
    limit_train: int | None,
    limit_valid: int | None,
    limit_test: int | None,
    empty_train_error: str,
) -> Mapping[str, list[genome.Interval]]:
    if not splits["train"]:
        raise ValueError(empty_train_error)

    _maybe_limit(splits["train"], limit_train)
    _maybe_limit(splits["valid"], limit_valid)
    _maybe_limit(splits["test"], limit_test)

    for key in list(splits.keys()):
        if not splits[key]:
            splits.pop(key)
    return splits


def load_intervals_from_dataframe(
    intervals_df: pd.DataFrame,
    window_size: int | None = None,
    *,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    limit_test: int | None = None,
) -> Mapping[str, list[genome.Interval]]:
    """Load ``train``/``valid``/``test`` intervals from a DataFrame.

    Args:
        intervals_df: Input table. If named columns ``chromosome``, ``start``,
            ``end``, ``split`` are present, they are used. Otherwise, the first
            four columns are interpreted in that order.
        window_size: Optional target window size; if set, intervals are recentered
            and resized via ``build_interval``.
        limit_train: Optional maximum number of train intervals to keep.
        limit_valid: Optional maximum number of valid intervals to keep.
        limit_test: Optional maximum number of test intervals to keep.

    Returns:
        Mapping from split name to list of ``genome.Interval`` objects.

    Raises:
        ValueError: If fewer than four columns are provided or no training
            intervals remain.
    """
    splits: dict[str, list[genome.Interval]] = defaultdict(list)
    required_cols = {"chromosome", "start", "end", "split"}
    has_named_cols = required_cols.issubset(set(intervals_df.columns))
    if not has_named_cols and intervals_df.shape[1] < 4:
        raise ValueError(
            "intervals_df must include columns {chromosome,start,end,split} "
            "or have at least 4 columns in that order."
        )

    if has_named_cols:
        row_iter = (
            (
                str(getattr(row, "chromosome")),
                int(float(getattr(row, "start"))),
                int(float(getattr(row, "end"))),
                str(getattr(row, "split")).lower(),
            )
            for row in intervals_df.itertuples(index=False)
        )
    else:
        row_iter = (
            (
                str(row[0]),
                int(float(row[1])),
                int(float(row[2])),
                str(row[3]).lower(),
            )
            for row in intervals_df.iloc[:, :4].itertuples(index=False, name=None)
        )

    for chrom, start, end, label in row_iter:
        if label not in {"train", "valid", "test"}:
            continue
        interval = build_interval(
            chromosome=chrom,
            start=start,
            end=end,
            window_size=window_size,
        )
        splits[label].append(interval)

    return _finalize_splits(
        splits,
        limit_train=limit_train,
        limit_valid=limit_valid,
        limit_test=limit_test,
        empty_train_error="No training intervals found in generated fold intervals.",
    )


def load_intervals_from_bed(
    bed_path: Path,
    window_size: int | None = None,
    *,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    limit_test: int | None = None,
) -> Mapping[str, list[genome.Interval]]:
    """Load ``train``/``valid``/``test`` intervals from a BED or BED.GZ file.

    The BED is expected to provide at least four fields per row:
    ``chrom start end split``. Rows with unknown split labels are skipped.

    Args:
        bed_path: Path to input BED/BED.GZ.
        window_size: Optional target window size; if set, intervals are recentered
            and resized via ``build_interval``.
        limit_train: Optional maximum number of train intervals to keep.
        limit_valid: Optional maximum number of valid intervals to keep.
        limit_test: Optional maximum number of test intervals to keep.

    Returns:
        Mapping from split name to list of ``genome.Interval`` objects.

    Raises:
        ValueError: If no training intervals are present after parsing/limits.
    """
    splits: dict[str, list[genome.Interval]] = defaultdict(list)
    opened = gzip.open if bed_path.suffix == ".gz" else open
    mode = "rt"
    with opened(bed_path, mode) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            chrom, start_str, end_str, split = parts[:4]
            label = split.lower()
            if label not in {"train", "valid", "test"}:
                continue
            interval = build_interval(
                chromosome=chrom,
                start=int(float(start_str)),
                end=int(float(end_str)),
                window_size=window_size,
            )
            splits[label].append(interval)

    return _finalize_splits(
        splits,
        limit_train=limit_train,
        limit_valid=limit_valid,
        limit_test=limit_test,
        empty_train_error="No training intervals found in --bed file.",
    )


def prepare_intervals_from_fold(
    fold: str | int,
    window_size: int | None = WINDOW_SIZE,
    organism: str = "HOMO_SAPIENS",
    *,
    bed_path: str | None = None,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    limit_test: int | None = None,
) -> Mapping[str, list[genome.Interval]]:
    """Build split intervals by deriving ``train``/``valid``/``test`` from a fold.

    Use this when you want to use the same fold-split as AlphaGenome.
    This function:
    1. Calls ``get_fold_split`` to map fold labels into ``train/valid/test``.
    2. Applies fold-generation logic (window construction + overlap filtering).
    3. Converts the generated split DataFrame into ``genome.Interval`` objects.

    Args:
        fold: Fold identifier (0, 1, 2, 3).
        window_size: Target fold window size. If ``None``, defaults to ``WINDOW_SIZE``.
        organism: Organism key/alias used for default fold BED and chromosome sizes.
        bed_path: Optional override for the fold BED/BED.GZ source.
        limit_train: Optional maximum number of train intervals.
        limit_valid: Optional maximum number of valid intervals.
        limit_test: Optional maximum number of test intervals.

    Returns:
        Mapping from split name to list of ``genome.Interval`` objects.

    Note:
        - Optionally, pass bed_path to override the default fold source.
        - ``prepare_intervals_from_split`` is different: it expects split labels to
          already exist in the input data (DataFrame/BED with ``split`` column).
    """
    fold_window_size = window_size if window_size is not None else WINDOW_SIZE
    split_df = get_fold_split(
        fold=fold,
        window_size=fold_window_size,
        organism=organism,
        bed_path=bed_path,
    )
    return load_intervals_from_dataframe(
        intervals_df=split_df,
        window_size=None,
        limit_train=limit_train,
        limit_valid=limit_valid,
        limit_test=limit_test,
    )


def prepare_intervals_from_split(
    intervals_df: pd.DataFrame | None = None,
    *,
    bed_path: Path | None = None,
    window_size: int | None = WINDOW_SIZE,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    limit_test: int | None = None,
) -> Mapping[str, list[genome.Interval]]:
    """Build split intervals from an input that already defines ``split`` labels.

    Use this when your input is already split into ``train/valid/test``:
    - ``intervals_df``: DataFrame with split labels (chr, start, end, and split columns).
    - ``bed_path``: BED/BED.GZ with at least ``chrom start end split`` columns.

    Exactly one source must be provided: ``intervals_df`` or ``bed_path``.

    Behavior:
    - Keeps only ``train``, ``valid``, and ``test`` rows.
    - Optionally recenters/resizes intervals via ``window_size``.
    - Applies optional per-split limits.

    Returns:
        Mapping from split name to list of ``genome.Interval`` objects.

    Note:
        ``prepare_intervals_from_fold`` is different: it derives split labels from
        fold labels using ``FOLD_MAPPING`` before interval conversion.
    """
    sources = (
        intervals_df is not None,
        bed_path is not None,
    )
    if sum(sources) != 1:
        raise ValueError(
            "prepare_intervals_from_split requires exactly one input source: "
            "one of intervals_df or bed_path."
        )

    if bed_path is not None:
        return load_intervals_from_bed(
            bed_path=bed_path,
            window_size=window_size,
            limit_train=limit_train,
            limit_valid=limit_valid,
            limit_test=limit_test,
        )

    return load_intervals_from_dataframe(
        intervals_df=intervals_df,
        window_size=window_size,
        limit_train=limit_train,
        limit_valid=limit_valid,
        limit_test=limit_test,
    )


class WindowedTargetCache:
    """Dense window-major binary cache for BigWig target values.

    Format:
      cache_dir/manifest.json
      cache_dir/<split>/<head_id>.npy

    Each ``.npy`` is a C-order array with shape
    ``[num_windows, window_size, num_tracks]`` and dtype float16 or float32.
    Missing/NaN BigWig values are stored as zero. ``manifest.json`` records the
    split intervals and source BigWig path, size, and mtime for validation.
    """

    FORMAT_VERSION = 1
    REPO_BRANCH_URL = "https://github.com/davidaknowles/alphagenome_ft/tree/fp4"

    def __init__(
        self,
        cache_dir: Path,
        *,
        intervals: Mapping[str, Sequence[genome.Interval]],
        head_specs: Sequence[HeadSpec],
        dtype: str | np.dtype = "float16",
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.dtype = _normalize_cache_dtype(dtype)
        self._intervals = {split: list(values) for split, values in intervals.items()}
        self._head_specs = list(head_specs)
        self._manifest_path = self.cache_dir / "manifest.json"
        self._arrays: dict[tuple[str, str], np.memmap] = {}

        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"Target cache manifest not found: {self._manifest_path}. "
                "Build it first with --build-target-cache."
            )
        self._manifest = _json_load(self._manifest_path)
        self._validate_manifest()

    @classmethod
    def build(
        cls,
        cache_dir: Path,
        *,
        intervals: Mapping[str, Sequence[genome.Interval]],
        head_specs: Sequence[HeadSpec],
        dtype: str | np.dtype = "float16",
        workers: int = 1,
        overwrite: bool = False,
    ) -> None:
        cache_dir = Path(cache_dir).expanduser().resolve()
        dtype_np = _normalize_cache_dtype(dtype)
        manifest_path = cache_dir / "manifest.json"
        manifest = cls._make_manifest(intervals, head_specs, dtype_np)

        if manifest_path.exists() and not overwrite:
            existing = _json_load(manifest_path)
            if existing == manifest:
                cls._write_readme(cache_dir, manifest)
                print(f"Target cache already exists and matches inputs: {cache_dir}")
                return
            raise ValueError(
                f"Target cache manifest already exists and does not match inputs: {manifest_path}. "
                "Use --overwrite-target-cache to rebuild."
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        _json_dump(manifest_path, manifest)

        effective_workers = max(1, int(workers))
        for split, split_intervals in intervals.items():
            if not split_intervals:
                continue
            split_dir = cache_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for spec in head_specs:
                output_path = split_dir / f"{spec.head_id}.npy"
                shape = (
                    len(split_intervals),
                    int(split_intervals[0].end - split_intervals[0].start),
                    len(spec.tracks),
                )
                if output_path.exists():
                    output_path.unlink()
                array = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=dtype_np,
                    shape=shape,
                )
                print(
                    "Building target cache: "
                    f"split={split}, head={spec.head_id}, shape={shape}, "
                    f"dtype={dtype_np.name}, path={output_path}"
                )
                cls._fill_array(array, split_intervals, spec, effective_workers)
                array.flush()
        cls._write_readme(cache_dir, manifest)

    @classmethod
    def _make_manifest(
        cls,
        intervals: Mapping[str, Sequence[genome.Interval]],
        head_specs: Sequence[HeadSpec],
        dtype: np.dtype,
    ) -> dict:
        return {
            "format": "alphagenome_ft.windowed_target_cache",
            "format_version": cls.FORMAT_VERSION,
            "dtype": dtype.name,
            "splits": {
                split: [_interval_record(interval) for interval in split_intervals]
                for split, split_intervals in intervals.items()
            },
            "heads": {
                spec.head_id: {
                    "tracks": [_track_record(Path(track.path)) for track in spec.tracks],
                }
                for spec in head_specs
            },
        }

    @classmethod
    def _write_readme(cls, cache_dir: Path, manifest: Mapping) -> None:
        split_lines = []
        for split, split_intervals in manifest["splits"].items():
            split_lines.append(f"- `{split}`: {len(split_intervals)} windows")

        head_lines = []
        for head_id, head_record in manifest["heads"].items():
            tracks = head_record["tracks"]
            head_lines.append(f"- `{head_id}`: {len(tracks)} tracks")
            if tracks:
                first_track = tracks[0]["path"]
                head_lines.append(f"  - first source track: `{first_track}`")

        readme = f"""# Alphagenome Windowed Target Cache

This directory contains a dense binary cache of BigWig targets for AlphaGenome
fine-tuning. It is generated by the `fp4` branch of this repository:

{cls.REPO_BRANCH_URL}

## Format

- Format name: `{manifest["format"]}`
- Format version: `{manifest["format_version"]}`
- Array dtype: `{manifest["dtype"]}`
- Arrays: `train/<head_id>.npy`, `valid/<head_id>.npy`, `test/<head_id>.npy`
- Array shape: `[num_windows, window_size, num_tracks]`
- Missing or NaN BigWig values are stored as `0.0`

## Splits

{chr(10).join(split_lines)}

## Heads

{chr(10).join(head_lines)}

## Manifest

`manifest.json` records every split interval and every source BigWig path,
file size, and mtime. The training loader validates this manifest before using
the cache, so moving or modifying the original BigWig files requires rebuilding
the cache.

## Rebuild

Use `scripts/slurm_build_humanbraindev_target_cache.sbatch` from the linked
branch, or run `scripts/run_humanbraindev_finetune.py` with
`--build-target-cache --target-cache-dir <this directory>`.
"""
        (cache_dir / "README.md").write_text(readme)

    @staticmethod
    def _fill_array(
        array: np.memmap,
        intervals: Sequence[genome.Interval],
        spec: HeadSpec,
        workers: int,
    ) -> None:
        if workers <= 1:
            handles = [pyBigWig.open(str(track.path)) for track in spec.tracks]
            try:
                for idx, interval in enumerate(intervals):
                    array[idx] = WindowedTargetCache._read_window(handles, interval, array.dtype)
                    if (idx + 1) % 100 == 0 or idx + 1 == len(intervals):
                        print(f"  cached {idx + 1}/{len(intervals)} windows", flush=True)
                return
            finally:
                for handle in handles:
                    handle.close()

        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        max_workers_by_fds = max(1, int((soft_limit - 64) // max(1, len(spec.tracks))))
        workers = min(workers, max_workers_by_fds)
        if workers < 1:
            workers = 1
        print(f"  cache workers={workers} (file descriptor limit={soft_limit})", flush=True)

        report_lock = threading.Lock()
        handle_lock = threading.Lock()
        thread_local = threading.local()
        all_thread_handles: list[list[pyBigWig.pyBigWig]] = []
        completed = 0

        def get_handles() -> list[pyBigWig.pyBigWig]:
            handles = getattr(thread_local, "handles", None)
            if handles is None:
                handles = [pyBigWig.open(str(track.path)) for track in spec.tracks]
                thread_local.handles = handles
                with handle_lock:
                    all_thread_handles.append(handles)
            return handles

        def build_one(index_interval: tuple[int, genome.Interval]):
            idx, interval = index_interval
            values = WindowedTargetCache._read_window(get_handles(), interval, array.dtype)
            return idx, values

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="alphagenome-cache",
            ) as executor:
                for idx, values in executor.map(build_one, enumerate(intervals)):
                    array[idx] = values
                    with report_lock:
                        completed += 1
                        if completed % 100 == 0 or completed == len(intervals):
                            print(f"  cached {completed}/{len(intervals)} windows", flush=True)
        finally:
            for handles in all_thread_handles:
                for handle in handles:
                    handle.close()

    @staticmethod
    def _read_window(
        handles: Sequence[pyBigWig.pyBigWig],
        interval: genome.Interval,
        dtype: np.dtype,
    ) -> np.ndarray:
        target_len = int(interval.end - interval.start)
        window = np.empty((target_len, len(handles)), dtype=dtype)
        for track_idx, handle in enumerate(handles):
            values = handle.values(interval.chromosome, interval.start, interval.end, numpy=True)
            arr = np.asarray(values, dtype=np.float32)
            np.nan_to_num(arr, copy=False, nan=0.0)
            if arr.shape[0] != target_len:
                padded = np.zeros((target_len,), dtype=np.float32)
                limit = min(target_len, arr.shape[0])
                padded[:limit] = arr[:limit]
                arr = padded
            window[:, track_idx] = arr.astype(dtype, copy=False)
        return window

    def _validate_manifest(self) -> None:
        expected = self._make_manifest(self._intervals, self._head_specs, self.dtype)
        if self._manifest != expected:
            raise ValueError(
                f"Target cache manifest does not match requested intervals/tracks: "
                f"{self._manifest_path}"
            )
        for split, split_intervals in self._intervals.items():
            for spec in self._head_specs:
                path = self.cache_dir / split / f"{spec.head_id}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Target cache array missing: {path}")
                expected_shape = (
                    len(split_intervals),
                    int(split_intervals[0].end - split_intervals[0].start),
                    len(spec.tracks),
                )
                actual = np.load(path, mmap_mode="r")
                if actual.shape != expected_shape or actual.dtype != self.dtype:
                    raise ValueError(
                        f"Target cache array has shape/dtype {actual.shape}/{actual.dtype}; "
                        f"expected {expected_shape}/{self.dtype.name}: {path}"
                    )

    def arrays_for_split(self, split: str) -> dict[str, np.memmap]:
        result: dict[str, np.memmap] = {}
        for spec in self._head_specs:
            key = (split, spec.head_id)
            if key not in self._arrays:
                self._arrays[key] = np.load(
                    self.cache_dir / split / f"{spec.head_id}.npy",
                    mmap_mode="r",
                )
            result[spec.head_id] = self._arrays[key]
        return result


class BigWigDataModule:
    """Creates training batches by streaming sequences + BigWig targets."""

    def __init__(
        self,
        *,
        intervals: Mapping[str, Sequence[genome.Interval]],
        fasta_path: Path,
        head_specs: Sequence[HeadSpec],
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        target_workers: int = 0,
        window_workers: int = 0,
        target_cache_dir: Path | None = None,
        target_cache_dtype: str | np.dtype = "float16",
        balance_gene_windows: bool = False,
        gene_window_repeats: int = 0,
    ) -> None:
        """Initialize streaming sequence/BigWig batch generation.

        Args:
            intervals: Split-to-interval mapping (``train``/``valid``/``test``).
            fasta_path: Reference FASTA used to extract sequence windows.
            head_specs: Head definitions including target BigWig track paths.
            batch_size: Number of windows per yielded batch.
            shuffle: Whether to shuffle window order in ``iter_batches``.
            drop_last: If True, drop incomplete final batches.
            target_workers: Number of host threads for reading BigWig target
                tracks within each window. ``0`` or ``1`` reads tracks serially.
            window_workers: Number of host threads for building windows within a
                batch. ``0`` or ``1`` keeps batch construction serial.
            target_cache_dir: Optional directory containing a windowed target
                cache built from these intervals and BigWig tracks.
            target_cache_dtype: dtype used by the target cache.
            balance_gene_windows: Distribute windows containing direct gene
                supervision across shuffled training batches without resampling.
            gene_window_repeats: Additional copies of each gene-bearing training
                window per epoch. The complete base window order is retained.

        Raises:
            ValueError: If no shared chromosomes exist across configured BigWigs,
                or no training intervals remain after chromosome filtering.
        """
        if target_workers < 0:
            raise ValueError(f"target_workers must be non-negative, got {target_workers}.")
        if window_workers < 0:
            raise ValueError(f"window_workers must be non-negative, got {window_workers}.")
        if gene_window_repeats < 0:
            raise ValueError("gene_window_repeats must be non-negative.")
        self._coverage_head_specs = tuple(
            spec
            for spec in head_specs
            if spec.gene_supervision_path is None or spec.coverage_loss_weight > 0
        )
        filtered_intervals = (
            self._filter_intervals_by_bigwig_chromosomes(intervals, self._coverage_head_specs)
            if self._coverage_head_specs
            else {split: list(values) for split, values in intervals.items()}
        )
        if not filtered_intervals.get("train"):
            raise ValueError(
                "No training intervals remain after filtering to chromosomes present in all BigWigs."
            )
        self._intervals = filtered_intervals
        self._fasta_path = fasta_path
        self._head_specs = head_specs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._target_workers = target_workers
        self._window_workers = window_workers
        self._balance_gene_windows = bool(balance_gene_windows)
        self._gene_window_repeats = int(gene_window_repeats)
        self._encoder = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)
        self._gene_supervisions = {
            spec.head_id: GeneExpressionSupervision(spec.gene_supervision_path, spec)
            for spec in head_specs
            if spec.gene_supervision_path is not None
        }
        if self._gene_window_repeats and not self._gene_supervisions:
            raise ValueError("gene_window_repeats requires direct gene supervision.")
        for supervision in self._gene_supervisions.values():
            supervision.configure_windows(self._intervals)
        self._max_genes = {
            head_name: supervision.max_genes(self._intervals)
            for head_name, supervision in self._gene_supervisions.items()
        }
        self._gene_counts_by_split = (
            {
                split: np.asarray(
                    [
                        sum(
                            len(supervision.indices_for_window(window))
                            for supervision in self._gene_supervisions.values()
                        )
                        for window in windows
                    ],
                    dtype=np.int64,
                )
                for split, windows in self._intervals.items()
            }
            if (self._balance_gene_windows or self._gene_window_repeats)
            and self._gene_supervisions
            else {}
        )
        for head_name, max_genes in self._max_genes.items():
            if max_genes == 0:
                raise ValueError(f"No genes were assigned to sequence windows for head {head_name}.")
        self._target_cache = (
            WindowedTargetCache(
                target_cache_dir,
                intervals=self._intervals,
                head_specs=self._coverage_head_specs,
                dtype=target_cache_dtype,
            )
            if target_cache_dir is not None and self._coverage_head_specs
            else None
        )

    def num_examples_per_epoch(self, split: str) -> int:
        count = len(self._intervals.get(split, ()))
        if split == "train" and self._gene_window_repeats and self._gene_counts_by_split:
            count += self._gene_window_repeats * int(
                np.count_nonzero(self._gene_counts_by_split[split])
            )
        return count

    def num_batches_per_epoch(self, split: str) -> int:
        count = self.num_examples_per_epoch(split)
        if self._drop_last:
            return count // self._batch_size
        return math.ceil(count / self._batch_size)

    @staticmethod
    def _get_common_bigwig_chromosomes(
        head_specs: Sequence[HeadSpec],
    ) -> set[str]:
        common_chroms: set[str] | None = None
        for spec in head_specs:
            for track in spec.tracks:
                with pyBigWig.open(str(track.path)) as handle:
                    chrom_dict = handle.chroms()
                if not chrom_dict:
                    raise ValueError(f"BigWig has no chromosome index: {track.path}")
                track_chroms = set(chrom_dict.keys())
                if common_chroms is None:
                    common_chroms = track_chroms
                else:
                    common_chroms &= track_chroms
                if not common_chroms:
                    return set()
        return common_chroms or set()

    @classmethod
    def _filter_intervals_by_bigwig_chromosomes(
        cls,
        intervals: Mapping[str, Sequence[genome.Interval]],
        head_specs: Sequence[HeadSpec],
    ) -> dict[str, list[genome.Interval]]:
        common_chroms = cls._get_common_bigwig_chromosomes(head_specs)
        if not common_chroms:
            raise ValueError("No shared chromosomes were found across configured BigWig tracks.")

        filtered: dict[str, list[genome.Interval]] = {}
        removed_chroms: set[str] = set()
        removed_counts: dict[str, int] = defaultdict(int)
        total_removed = 0

        for split, split_intervals in intervals.items():
            kept: list[genome.Interval] = []
            for interval in split_intervals:
                if interval.chromosome in common_chroms:
                    kept.append(interval)
                else:
                    removed_chroms.add(interval.chromosome)
                    removed_counts[split] += 1
                    total_removed += 1
            if kept:
                filtered[split] = kept

        if total_removed > 0:
            removed = ", ".join(sorted(removed_chroms))
            split_counts = ", ".join(
                f"{split}={removed_counts[split]}"
                for split in ("train", "valid", "test")
                if removed_counts.get(split, 0) > 0
            )
            print(
                "Filtered out "
                f"{total_removed} intervals to match BigWig chromosome coverage "
                f"(removed chromosomes: {removed}; removed intervals: {split_counts})."
            )

        return filtered

    def iter_batches(
        self, split: str, *, seed: int | None = None, shuffle: bool | None = None
    ) -> Iterator[dict[str, np.ndarray]]:
        windows = list(self._intervals.get(split, ()))
        if not windows:
            return

        order = np.arange(len(windows))
        should_shuffle = self._shuffle if shuffle is None else shuffle
        rng = None
        if should_shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        if split == "train" and self._gene_window_repeats:
            order = _repeat_gene_window_order(
                order,
                self._gene_counts_by_split[split],
                additional_repeats=self._gene_window_repeats,
                rng=rng,
            )
        if should_shuffle:
            if self._balance_gene_windows and self._gene_supervisions:
                order = _balance_gene_window_order(
                    order,
                    self._gene_counts_by_split[split],
                    batch_size=self._batch_size,
                    drop_last=self._drop_last,
                )

        extractor = fasta_lib.FastaExtractor(str(self._fasta_path))
        with contextlib.ExitStack() as stack:
            target_cache_arrays = (
                self._target_cache.arrays_for_split(split)
                if self._target_cache is not None
                else None
            )
            head_handles: dict[str, list[pyBigWig.pyBigWig]] = {}
            if target_cache_arrays is None:
                for spec in self._coverage_head_specs:
                    handles = []
                    for track in spec.tracks:
                        handles.append(stack.enter_context(pyBigWig.open(str(track.path))))
                    head_handles[spec.head_id] = handles
            available_cpus = _available_cpu_count()
            effective_target_workers = min(self._target_workers, available_cpus)
            target_executor = None
            if target_cache_arrays is None and effective_target_workers > 1:
                target_executor = stack.enter_context(
                    concurrent.futures.ThreadPoolExecutor(
                        max_workers=effective_target_workers,
                        thread_name_prefix="alphagenome-bigwig",
                    )
                )
            window_executor = (
                stack.enter_context(
                    concurrent.futures.ThreadPoolExecutor(
                        max_workers=self._window_workers,
                        thread_name_prefix="alphagenome-window",
                    )
                )
                if self._window_workers > 1
                else None
            )

            batch_indices: list[int] = []
            for idx in order:
                batch_indices.append(int(idx))
                if len(batch_indices) == self._batch_size:
                    if window_executor is None:
                        yield self._make_batch(
                            batch_indices,
                            windows,
                            extractor,
                            head_handles,
                            target_executor,
                            target_cache_arrays,
                        )
                    else:
                        yield self._make_batch_parallel(
                            batch_indices,
                            windows,
                            head_handles,
                            window_executor,
                            target_executor,
                            target_cache_arrays,
                        )
                    batch_indices = []

            if batch_indices and not self._drop_last:
                if window_executor is None:
                    yield self._make_batch(
                        batch_indices,
                        windows,
                        extractor,
                        head_handles,
                        target_executor,
                        target_cache_arrays,
                    )
                else:
                    yield self._make_batch_parallel(
                        batch_indices,
                        windows,
                        head_handles,
                        window_executor,
                        target_executor,
                        target_cache_arrays,
                    )

    def _make_batch(
        self,
        batch_indices: Sequence[int],
        windows: Sequence[genome.Interval],
        extractor: fasta_lib.FastaExtractor,
        head_handles: Mapping[str, Sequence[pyBigWig.pyBigWig]],
        target_executor: concurrent.futures.Executor | None = None,
        target_cache_arrays: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        sequences = []
        targets: dict[str, list[np.ndarray]] = {
            spec.head_id: [] for spec in self._coverage_head_specs
        }

        for idx in batch_indices:
            window = windows[idx]
            seq = extractor.extract(window)
            encoded = self._encoder.encode(seq)
            sequences.append(encoded)

            seq_len = encoded.shape[0]
            for spec in self._coverage_head_specs:
                if target_cache_arrays is not None:
                    channel_arrays = target_cache_arrays[spec.head_id][idx]
                    targets[spec.head_id].append(np.asarray(channel_arrays))
                    continue
                handles = head_handles[spec.head_id]
                if target_executor is None:
                    channel_arrays = [
                        self._read_track(handle, window, seq_len) for handle in handles
                    ]
                else:
                    channel_arrays = list(
                        target_executor.map(
                            lambda handle: self._read_track(handle, window, seq_len),
                            handles,
                        )
                    )
                targets[spec.head_id].append(np.stack(channel_arrays, axis=-1))

        batch = {
            "sequences": np.stack(sequences, axis=0).astype(np.float32),
            "negative_strand_mask": np.zeros((len(batch_indices),), dtype=bool),
        }
        for head_name, arrays in targets.items():
            if target_cache_arrays is None:
                batch[f"targets_{head_name}"] = np.stack(arrays, axis=0).astype(np.float32)
            else:
                batch[f"targets_{head_name}"] = np.stack(arrays, axis=0)
        for head_name, supervision in self._gene_supervisions.items():
            gene_arrays = [
                supervision.arrays_for_window(
                    windows[idx],
                    sequence_length=batch["sequences"].shape[1],
                    max_genes=self._max_genes[head_name],
                )
                for idx in batch_indices
            ]
            for key in ("weights", "targets", "strands", "valid"):
                batch[f"gene_{key}_{head_name}"] = np.stack(
                    [arrays[key] for arrays in gene_arrays], axis=0
                )
        return batch

    def _make_batch_parallel(
        self,
        batch_indices: Sequence[int],
        windows: Sequence[genome.Interval],
        head_handles: Mapping[str, Sequence[pyBigWig.pyBigWig]],
        window_executor: concurrent.futures.Executor,
        target_executor: concurrent.futures.Executor | None = None,
        target_cache_arrays: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        thread_local = threading.local()

        def get_extractor() -> fasta_lib.FastaExtractor:
            extractor = getattr(thread_local, "extractor", None)
            if extractor is None:
                extractor = fasta_lib.FastaExtractor(str(self._fasta_path))
                thread_local.extractor = extractor
            return extractor

        def get_encoder() -> one_hot_encoder.DNAOneHotEncoder:
            encoder = getattr(thread_local, "encoder", None)
            if encoder is None:
                encoder = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)
                thread_local.encoder = encoder
            return encoder

        def build_sequence(idx: int) -> np.ndarray:
            extractor = get_extractor()
            encoder = get_encoder()
            window = windows[idx]
            seq = extractor.extract(window)
            return encoder.encode(seq).astype(np.float32)

        sequences = list(window_executor.map(build_sequence, batch_indices))
        sequence_length = sequences[0].shape[0]
        targets: dict[str, list[np.ndarray]] = {
            spec.head_id: [] for spec in self._coverage_head_specs
        }
        for spec in self._coverage_head_specs:
            if target_cache_arrays is not None:
                targets[spec.head_id] = [
                    np.asarray(target_cache_arrays[spec.head_id][idx]) for idx in batch_indices
                ]
                continue

            def read_track_batch(handle):
                return [
                    self._read_track(handle, windows[idx], sequence_length) for idx in batch_indices
                ]

            handles = head_handles[spec.head_id]
            if target_executor is None:
                arrays_by_track = [read_track_batch(handle) for handle in handles]
            else:
                arrays_by_track = list(target_executor.map(read_track_batch, handles))
            targets[spec.head_id] = [
                np.stack(
                    [track_arrays[window_index] for track_arrays in arrays_by_track],
                    axis=-1,
                ).astype(np.float32)
                for window_index in range(len(batch_indices))
            ]

        batch = {
            "sequences": np.stack(sequences, axis=0),
            "negative_strand_mask": np.zeros((len(batch_indices),), dtype=bool),
        }
        for head_name, arrays in targets.items():
            batch[f"targets_{head_name}"] = np.stack(arrays, axis=0)
        for head_name, supervision in self._gene_supervisions.items():
            gene_arrays = [
                supervision.arrays_for_window(
                    windows[idx],
                    sequence_length=batch["sequences"].shape[1],
                    max_genes=self._max_genes[head_name],
                )
                for idx in batch_indices
            ]
            for key in ("weights", "targets", "strands", "valid"):
                batch[f"gene_{key}_{head_name}"] = np.stack(
                    [arrays[key] for arrays in gene_arrays], axis=0
                )
        return batch

    def _read_track(
        self,
        handle: pyBigWig.pyBigWig,
        window: genome.Interval,
        target_len: int,
    ) -> np.ndarray:
        values = handle.values(window.chromosome, window.start, window.end, numpy=True)
        return self._prepare_track(values, target_len)

    @staticmethod
    def _prepare_track(values: Sequence[float] | None, target_len: int) -> np.ndarray:
        if values is None:
            padded = np.zeros((target_len,), dtype=np.float32)
            return padded
        arr = np.asarray(values, dtype=np.float32)
        np.nan_to_num(arr, copy=False, nan=0.0)
        if arr.shape[0] == target_len:
            return arr
        padded = np.zeros((target_len,), dtype=np.float32)
        limit = min(target_len, arr.shape[0])
        padded[:limit] = arr[:limit]
        return padded


class MultiSpeciesDataModule:
    """Round-robin batches from species-specific modules with shared heads."""

    def __init__(
        self,
        modules: Mapping[str, BigWigDataModule],
        *,
        organism_indices: Mapping[str, int] | None = None,
    ) -> None:
        if not modules:
            raise ValueError("At least one species data module is required.")
        self._modules = dict(modules)
        self._organism_indices = dict(organism_indices or {})
        unknown_species = set(self._organism_indices) - set(self._modules)
        if unknown_species:
            raise ValueError(f"Organism indices contain unknown species: {sorted(unknown_species)}")
        drop_last_values = {module._drop_last for module in self._modules.values()}
        if len(drop_last_values) != 1:
            raise ValueError("All species data modules must use the same drop_last setting.")
        self._drop_last = drop_last_values.pop()
        batch_sizes = {module._batch_size for module in self._modules.values()}
        if len(batch_sizes) != 1:
            raise ValueError("All species data modules must use the same batch size.")
        self._batch_size = batch_sizes.pop()

        all_gene_heads = {
            head_name for module in self._modules.values() for head_name in module._max_genes
        }
        for head_name in all_gene_heads:
            global_max = max(
                module._max_genes.get(head_name, 0) for module in self._modules.values()
            )
            for species, module in self._modules.items():
                if head_name not in module._max_genes:
                    raise ValueError(
                        f"Species {species} is missing gene supervision for {head_name}."
                    )
                module._max_genes[head_name] = global_max

        self._intervals: dict[str, list[genome.Interval]] = {}
        split_names = set.intersection(
            *(set(module._intervals) for module in self._modules.values())
        )
        for split in split_names:
            species_intervals = [module._intervals[split] for module in self._modules.values()]
            if split == "train":
                common_count = min(len(intervals) for intervals in species_intervals)
                if self._drop_last:
                    batch_size = next(iter(self._modules.values()))._batch_size
                    common_count = common_count // batch_size * batch_size
                self._intervals[split] = [
                    interval
                    for intervals in species_intervals
                    for interval in intervals[:common_count]
                ]
            else:
                self._intervals[split] = [
                    interval for intervals in species_intervals for interval in intervals
                ]

    @staticmethod
    def _module_batch_count(module: BigWigDataModule, split: str) -> int:
        if hasattr(module, "num_batches_per_epoch"):
            return module.num_batches_per_epoch(split)
        interval_count = len(module._intervals.get(split, ()))
        if module._drop_last:
            return interval_count // module._batch_size
        return math.ceil(interval_count / module._batch_size)

    def num_batches_per_epoch(self, split: str) -> int:
        counts = [
            self._module_batch_count(module, split) for module in self._modules.values()
        ]
        return min(counts) * len(counts) if split == "train" else sum(counts)

    def num_examples_per_epoch(self, split: str) -> int:
        return self.num_batches_per_epoch(split) * self._batch_size

    def iter_batches(
        self,
        split: str,
        *,
        seed: int | None = None,
        shuffle: bool | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        def tagged_batches(species: str, module: BigWigDataModule, species_idx: int):
            for batch in module.iter_batches(
                split,
                seed=None if seed is None else seed + species_idx,
                shuffle=shuffle,
            ):
                organism_index = self._organism_indices.get(species)
                if organism_index is not None:
                    batch = dict(batch)
                    batch["organism_index"] = np.full(
                        (batch["sequences"].shape[0],), organism_index, dtype=np.int32
                    )
                yield batch

        iterators = [
            iter(tagged_batches(species, module, species_idx))
            for species_idx, (species, module) in enumerate(self._modules.items())
        ]
        if split == "train":
            batch_counts = [
                self._module_batch_count(module, split)
                for module in self._modules.values()
            ]
            for _ in range(min(batch_counts)):
                for iterator in iterators:
                    yield next(iterator)
            return
        active = list(range(len(iterators)))
        while active:
            next_active: list[int] = []
            for species_idx in active:
                try:
                    yield next(iterators[species_idx])
                    next_active.append(species_idx)
                except StopIteration:
                    pass
            active = next_active


def build_fasta_index(fasta_path: Path) -> dict[str, int]:
    """Ensure ``<fasta>.fai`` exists and return chromosome sizes from the index.

    Args:
        fasta_path: Path to an uncompressed reference FASTA (for example,
            ``hg38.fa``).

    Raises:
        FileNotFoundError: If ``fasta_path`` does not exist.
        ValueError: If input is gzipped FASTA or index contents are invalid.
        ModuleNotFoundError: If ``pyfaidx`` is unavailable when index creation
            is needed.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if fasta_path.suffix == ".gz":
        raise ValueError(
            f"Expected uncompressed FASTA, got gzipped file: {fasta_path}. "
            "Please decompress before indexing."
        )

    fai_path = Path(f"{fasta_path}.fai")
    if not fai_path.exists():
        try:
            from pyfaidx import Fasta  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyfaidx is required to build FASTA indexes. Install `pyfaidx` "
                "or provide an existing .fai file."
            ) from exc

        fasta = Fasta(str(fasta_path))
        fasta.close()

    if not fai_path.exists():
        raise FileNotFoundError(f"Failed to create FASTA index: {fai_path}")

    chromosome_sizes: dict[str, int] = {}
    with fai_path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"Invalid FASTA index row in {fai_path}: {line!r}")
            chromosome_sizes[fields[0]] = int(fields[1])
    if not chromosome_sizes:
        raise ValueError(f"FASTA index contains no chromosomes: {fai_path}")
    return chromosome_sizes


def prepare_batch(
    batch: Mapping[str, np.ndarray],
    organism_index_value: int,
    head_names: Sequence[str],
):
    """Convert a numpy batch to JAX arrays and attach organism/head fields.

    Args:
        batch: Batch mapping containing ``sequences``, ``negative_strand_mask``,
            and per-head arrays under ``targets_{head_name}``.
        organism_index_value: Integer organism index to broadcast across batch.
        head_names: Head names to extract from ``batch`` target keys.

    Returns:
        Mapping ready for model calls with JAX arrays:
        ``sequences``, ``organism_index``, ``negative_strand_mask``, and
        ``targets_{head_name}`` for each head.
    """
    import jax.numpy as jnp

    organism_index = batch.get("organism_index")
    if organism_index is None:
        organism_index = np.full(
            (batch["sequences"].shape[0],), organism_index_value, dtype=np.int32
        )
    elif np.asarray(organism_index).shape != (batch["sequences"].shape[0],):
        raise ValueError("Batch organism_index must contain one value per sequence.")
    prepared = {
        "sequences": jnp.asarray(batch["sequences"]),
        "organism_index": jnp.asarray(organism_index, dtype=jnp.int32),
        "negative_strand_mask": jnp.asarray(batch["negative_strand_mask"]),
    }
    for head_name in head_names:
        target_key = f"targets_{head_name}"
        if target_key in batch:
            prepared[target_key] = jnp.asarray(batch[target_key])
        for key in ("weights", "targets", "strands", "valid"):
            batch_key = f"gene_{key}_{head_name}"
            if batch_key in batch:
                prepared[batch_key] = jnp.asarray(batch[batch_key])
    return prepared
