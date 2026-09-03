"""Build binned ATAC targets from paired-fragment records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Collection, Mapping, Sequence

import h5py
import numpy as np

from alphagenome_ft.finetune.reliability import balanced_group_item_split
from alphagenome_ft.finetune.rna_tracks import _read_h5ad_column


def _decode(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    return str(value)


def read_cell_groups(
    h5ad_path: Path,
    *,
    index_column: str = "_index",
    group_column: str = "Group",
) -> dict[str, str]:
    """Read a cell identifier to group mapping without loading the expression matrix."""
    with h5py.File(Path(h5ad_path), "r") as handle:
        obs = handle["obs"]
        indices = _read_h5ad_column(obs, index_column)
        groups = _read_h5ad_column(obs, group_column)
    if len(indices) != len(groups):
        raise ValueError(
            f"Cell index and group columns have different lengths, {len(indices)} and {len(groups)}."
        )
    result = {_decode(index): _decode(group) for index, group in zip(indices, groups)}
    if len(result) != len(indices):
        raise ValueError(f"Cell identifiers in {h5ad_path} are not unique.")
    if any(not group for group in result.values()):
        raise ValueError(f"Cell groups in {h5ad_path} must be nonempty.")
    return result


def read_cell_groups_by_library(
    h5ad_path: Path,
    *,
    barcode_column: str = "cell_barcode",
    library_column: str = "barcoded_cell_sample_label",
    group_column: str = "Group",
) -> dict[str, dict[str, str]]:
    """Read library-local cell barcodes and their target groups from an H5AD file."""
    with h5py.File(Path(h5ad_path), "r") as handle:
        obs = handle["obs"]
        barcodes = _read_h5ad_column(obs, barcode_column)
        libraries = _read_h5ad_column(obs, library_column)
        groups = _read_h5ad_column(obs, group_column)
    if not (len(barcodes) == len(libraries) == len(groups)):
        raise ValueError("Cell barcode, library, and group columns must have equal lengths.")

    result: dict[str, dict[str, str]] = {}
    for barcode_value, library_value, group_value in zip(barcodes, libraries, groups):
        barcode = _decode(barcode_value)
        library = _decode(library_value)
        group = _decode(group_value)
        library_groups = result.setdefault(library, {})
        previous = library_groups.setdefault(barcode, group)
        if previous != group:
            raise ValueError(f"Barcode {barcode!r} maps to multiple groups in library {library!r}.")
    return result


def _normalize_10x_barcode(cell: str) -> str:
    return cell[:-2] if cell.endswith("-1") else cell


def match_fragment_library(
    fragment_cells: Collection[str],
    cell_groups_by_library: Mapping[str, Mapping[str, str]],
) -> tuple[str, dict[str, str]]:
    """Match library-local 10x fragment barcodes to one metadata library."""
    if not fragment_cells:
        raise ValueError("Fragment cell identifiers must be nonempty.")
    normalized_cells = {cell: _normalize_10x_barcode(cell) for cell in fragment_cells}
    scores = {
        library: sum(barcode in cell_groups for barcode in normalized_cells.values())
        for library, cell_groups in cell_groups_by_library.items()
    }
    best_score = max(scores.values(), default=0)
    winners = [library for library, score in scores.items() if score == best_score]
    if best_score != len(normalized_cells) or len(winners) != 1:
        raise ValueError(
            "Fragment barcodes must have one unique metadata library with complete coverage, "
            f"found best score {best_score}/{len(normalized_cells)} across {winners}."
        )
    library = winners[0]
    library_groups = cell_groups_by_library[library]
    return library, {cell: library_groups[barcode] for cell, barcode in normalized_cells.items()}


@dataclass
class BinnedAtacAccumulator:
    """Accumulate insertion counts and fragment-covered bases by group and bin."""

    num_groups: int
    chromosome_size: int
    bin_size: int = 100
    tn5_shift: bool = True

    def __post_init__(self) -> None:
        if self.num_groups <= 0 or self.chromosome_size <= 0 or self.bin_size <= 0:
            raise ValueError("Group count, chromosome size, and bin size must be positive.")
        self.num_bins = (self.chromosome_size + self.bin_size - 1) // self.bin_size
        self.insertion_counts = np.zeros((self.num_groups, self.num_bins), dtype=np.float64)
        self.covered_bases = np.zeros((self.num_groups, self.num_bins), dtype=np.float64)
        self._coverage_difference = np.zeros((self.num_groups, self.num_bins + 1), dtype=np.float64)
        self.fragment_counts = np.zeros(self.num_groups, dtype=np.float64)
        self.records = 0

    def add(
        self,
        group_indices: Sequence[int],
        starts: Sequence[int],
        ends: Sequence[int],
        counts: Sequence[int] | None = None,
    ) -> None:
        groups = np.asarray(group_indices, dtype=np.int64)
        starts_array = np.asarray(starts, dtype=np.int64)
        ends_array = np.asarray(ends, dtype=np.int64)
        if counts is None:
            counts_array = np.ones(groups.shape, dtype=np.float64)
        else:
            counts_array = np.asarray(counts, dtype=np.float64)
        if not (groups.shape == starts_array.shape == ends_array.shape == counts_array.shape):
            raise ValueError("Groups, starts, ends, and counts must have the same shape.")

        valid = (
            (groups >= 0)
            & (groups < self.num_groups)
            & (starts_array >= 0)
            & (ends_array > starts_array)
            & (ends_array <= self.chromosome_size)
            & (counts_array > 0)
        )
        if not np.all(valid):
            raise ValueError(f"Received {np.count_nonzero(~valid)} invalid fragment record(s).")
        if groups.size == 0:
            return

        np.add.at(self.fragment_counts, groups, counts_array)
        self.records += int(groups.size)

        left_cuts = starts_array + (4 if self.tn5_shift else 0)
        right_cuts = ends_array - (5 if self.tn5_shift else 1)
        left_cuts = np.clip(left_cuts, 0, self.chromosome_size - 1)
        right_cuts = np.clip(right_cuts, 0, self.chromosome_size - 1)
        np.add.at(
            self.insertion_counts,
            (groups, left_cuts // self.bin_size),
            counts_array,
        )
        np.add.at(
            self.insertion_counts,
            (groups, right_cuts // self.bin_size),
            counts_array,
        )

        first_bins = starts_array // self.bin_size
        last_bins = (ends_array - 1) // self.bin_size
        same_bin = first_bins == last_bins
        if np.any(same_bin):
            np.add.at(
                self.covered_bases,
                (groups[same_bin], first_bins[same_bin]),
                (ends_array[same_bin] - starts_array[same_bin]) * counts_array[same_bin],
            )

        multiple_bins = ~same_bin
        if np.any(multiple_bins):
            multi_groups = groups[multiple_bins]
            multi_counts = counts_array[multiple_bins]
            multi_starts = starts_array[multiple_bins]
            multi_ends = ends_array[multiple_bins]
            multi_first = first_bins[multiple_bins]
            multi_last = last_bins[multiple_bins]
            np.add.at(
                self.covered_bases,
                (multi_groups, multi_first),
                ((multi_first + 1) * self.bin_size - multi_starts) * multi_counts,
            )
            np.add.at(
                self.covered_bases,
                (multi_groups, multi_last),
                (multi_ends - multi_last * self.bin_size) * multi_counts,
            )
            np.add.at(
                self._coverage_difference,
                (multi_groups, multi_first + 1),
                multi_counts,
            )
            np.add.at(
                self._coverage_difference,
                (multi_groups, multi_last),
                -multi_counts,
            )

    def finalize(self) -> None:
        if self._coverage_difference is None:
            return
        self.covered_bases += np.cumsum(self._coverage_difference[:, :-1], axis=1) * self.bin_size
        self._coverage_difference = None

    def normalized(
        self,
        total_fragments_by_group: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return insertion and mean fragment-depth signals per million fragments."""
        self.finalize()
        totals = np.asarray(total_fragments_by_group, dtype=np.float64)
        if totals.shape != (self.num_groups,) or np.any(totals <= 0):
            raise ValueError(
                f"Expected {self.num_groups} positive group totals, received shape {totals.shape}."
            )
        insertion_spmr = self.insertion_counts / (totals[:, None] / 1_000_000.0)
        widths = np.minimum(
            self.bin_size,
            self.chromosome_size - np.arange(self.num_bins) * self.bin_size,
        )
        coverage_spmr = self.covered_bases / widths[None, :] / (totals[:, None] / 1_000_000.0)
        return insertion_spmr.astype(np.float32), coverage_spmr.astype(np.float32)


def fragment_totals_by_group(
    per_cell_counts: Mapping[str, int],
    cell_groups: Mapping[str, str],
    groups: Sequence[str],
) -> tuple[np.ndarray, int]:
    """Sum whole-genome fragment multiplicities by group."""
    group_indices = {group: index for index, group in enumerate(groups)}
    totals = np.zeros(len(groups), dtype=np.float64)
    missing = 0
    for cell, count in per_cell_counts.items():
        group = cell_groups.get(cell)
        if group is None:
            missing += 1
            continue
        totals[group_indices[group]] += int(count)
    return totals, missing


def depth_balanced_half_assignments(
    fragment_paths: Sequence[Path],
    histograms: Mapping[Path, Mapping[str, int]],
    groups_by_path: Mapping[Path, Mapping[str, str]],
) -> tuple[list[str], dict[Path, dict[str, int]], np.ndarray, int]:
    """Split cells into deterministic depth-balanced halves within every group.

    Fragment multiplicities balance and normalize whole-cell pseudobulk halves;
    each cell remains in one half, preserving a biological rather than technical split.
    """
    group_names = sorted({group for groups in groups_by_path.values() for group in groups.values()})
    group_index = {group: index for index, group in enumerate(group_names)}
    items_by_group: dict[str, list[tuple[Path, str, int]]] = {group: [] for group in group_names}
    histogram_cells_missing_metadata = 0
    for path in fragment_paths:
        cell_groups = groups_by_path[path]
        for cell, count in histograms[path].items():
            group = cell_groups.get(cell)
            if group is None:
                histogram_cells_missing_metadata += 1
                continue
            items_by_group[group].append((path, cell, count))

    assignments: dict[Path, dict[str, int]] = {path: {} for path in fragment_paths}
    half_depths = np.zeros((len(group_names), 2), dtype=np.float64)
    for group, items in items_by_group.items():
        if not items:
            continue
        item_groups = np.repeat(group, len(items))
        weights = np.asarray([item[2] for item in items], dtype=np.float64)
        keys = np.asarray([f"{item[0]}\0{item[1]}" for item in items])
        first_half = balanced_group_item_split(item_groups, weights, keys)
        group_offset = 2 * group_index[group]
        for item, first in zip(items, first_half, strict=True):
            path, cell, count = item
            half = 0 if first else 1
            assignments[path][cell] = group_offset + half
            half_depths[group_index[group], half] += count
    return group_names, assignments, half_depths, histogram_cells_missing_metadata


def read_fragment_histogram(path: Path) -> dict[str, int]:
    """Read a GRR per-cell whole-genome fragment histogram."""
    payload = json.loads(Path(path).read_text())
    values = payload.get("values")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"Missing per-cell fragment counts in {path}.")
    return {str(cell): int(count) for cell, count in values.items()}


def stream_tabix_fragments(
    path: Path,
    chromosome: str,
    *,
    cell_group_indices: Mapping[str, int],
    accumulator: BinnedAtacAccumulator,
    chunk_size: int = 250_000,
    tabix: str = "tabix",
) -> tuple[int, int]:
    """Stream one chromosome from a tabix-indexed fragment file into an accumulator."""
    process = subprocess.Popen(
        [tabix, str(path), chromosome],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    group_indices: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    counts: list[int] = []
    matched = 0
    missing = 0

    def add_chunk() -> None:
        accumulator.add(group_indices, starts, ends, counts)
        group_indices.clear()
        starts.clear()
        ends.clear()
        counts.clear()

    for line in process.stdout:
        fields = line.rstrip().split("\t")
        if len(fields) < 5:
            process.kill()
            raise ValueError(f"Malformed fragment row in {path}, {line[:200]!r}")
        group_index = cell_group_indices.get(fields[3])
        if group_index is None:
            missing += 1
            continue
        group_indices.append(group_index)
        starts.append(int(fields[1]))
        ends.append(int(fields[2]))
        counts.append(int(fields[4]))
        matched += 1
        if len(group_indices) >= chunk_size:
            add_chunk()
    if group_indices:
        add_chunk()
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"tabix failed for {path} with code {return_code}, {stderr.strip()}")
    return matched, missing


__all__ = [
    "BinnedAtacAccumulator",
    "fragment_totals_by_group",
    "read_fragment_histogram",
    "read_cell_groups",
    "stream_tabix_fragments",
]
