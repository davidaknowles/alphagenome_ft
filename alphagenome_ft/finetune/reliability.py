"""Reliability utilities for replicated pseudobulk targets."""

from __future__ import annotations

import numpy as np


def balanced_library_split(library_sizes: np.ndarray) -> np.ndarray:
    """Assign samples to two greedily depth-balanced halves for every group.

    ``library_sizes`` has shape ``[S, C]``, where ``S`` is the number of
    biological samples and ``C`` is the number of target cell groups. The
    returned Boolean matrix has the same shape and marks the first half.
    """
    library_sizes = np.asarray(library_sizes, dtype=np.float64)
    if library_sizes.ndim != 2 or np.any(~np.isfinite(library_sizes)):
        raise ValueError("library_sizes must be a finite [samples, groups] matrix.")
    if np.any(library_sizes < 0):
        raise ValueError("library_sizes must be non-negative.")
    first_half = np.zeros(library_sizes.shape, dtype=bool)
    for group in range(library_sizes.shape[1]):
        sample_indices = np.flatnonzero(library_sizes[:, group] > 0)
        ordered = sample_indices[np.argsort(-library_sizes[sample_indices, group])]
        totals = [0.0, 0.0]
        for sample in ordered:
            half = int(totals[1] < totals[0])
            first_half[sample, group] = half == 0
            totals[half] += library_sizes[sample, group]
    return first_half


def counts_per_million(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a ``[C, G]`` count matrix and return valid group rows."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite non-negative [groups, genes] matrix.")
    depths = counts.sum(axis=1)
    valid = depths > 0
    normalized = np.zeros_like(counts)
    normalized[valid] = counts[valid] / depths[valid, None] * 1_000_000.0
    return normalized, valid


def binomial_count_split(
    counts: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly divide each observed molecule between two technical halves.

    ``counts`` has shape ``[C, G]``, where ``C`` is the number of target cell
    groups and ``G`` is the number of genes. The two returned integer matrices
    have the same shape and sum exactly to the input counts.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite non-negative [groups, genes] matrix.")
    rounded = np.rint(counts)
    if not np.array_equal(counts, rounded):
        raise ValueError("counts must contain integer-valued molecule counts.")
    integer_counts = rounded.astype(np.int64, copy=False)
    first = np.random.default_rng(seed).binomial(integer_counts, 0.5)
    return first, integer_counts - first


def double_centered_pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate signed Pearson correlation after centering both matrix axes."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Inputs must be equal-shape [observations, tracks] matrices.")

    def center(values: np.ndarray) -> np.ndarray:
        return values - values.mean(axis=0) - values.mean(axis=1)[:, None] + values.mean()

    first = center(first)
    second = center(second)
    denominator = np.sqrt(np.sum(first * first) * np.sum(second * second))
    return float(np.sum(first * second) / denominator) if denominator > 0 else float("nan")


def spearman_brown(split_half_correlation: float) -> float:
    """Estimate full-target reliability from equal-half correlation."""
    correlation = float(split_half_correlation)
    return 2.0 * correlation / (1.0 + correlation)


def split_half_pseudobulks(
    counts_by_sample_group: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct depth-balanced sample split halves for pseudobulk counts.

    ``counts_by_sample_group`` has shape ``[S, C, G]``, where ``S`` is the
    number of biological samples, ``C`` is the number of cell groups, and
    ``G`` is the number of genes. Returned CPM matrices have shape ``[C, G]``;
    the Boolean vector of shape ``[C]`` marks groups observed in both halves.
    """
    counts = np.asarray(counts_by_sample_group, dtype=np.float64)
    if counts.ndim != 3 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts_by_sample_group must be finite non-negative [S, C, G].")
    library_sizes = counts.sum(axis=2)
    first_assignment = balanced_library_split(library_sizes)
    first_counts = np.sum(counts * first_assignment[..., None], axis=0)
    second_counts = np.sum(counts * (~first_assignment)[..., None], axis=0)
    first_cpm, first_valid = counts_per_million(first_counts)
    second_cpm, second_valid = counts_per_million(second_counts)
    return first_cpm, second_cpm, first_valid & second_valid
