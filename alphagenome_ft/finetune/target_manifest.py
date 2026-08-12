"""Utilities for constructing validated BigWig target manifests."""

from __future__ import annotations

import copy
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyBigWig


def combined_bigwig_nonzero_mean(paths: Sequence[Path]) -> float:
    """Return the positive-base mean after summing aligned BigWig tracks."""
    paths = tuple(Path(path).expanduser() for path in paths)
    if not paths:
        raise ValueError("At least one BigWig is required.")
    handles = [pyBigWig.open(str(path)) for path in paths]
    weighted_sum = 0.0
    positive_bases = 0
    try:
        chromosome_sizes = handles[0].chroms()
        if any(handle.chroms() != chromosome_sizes for handle in handles[1:]):
            raise ValueError("Combined BigWigs must have identical chromosome headers.")
        for chromosome in chromosome_sizes:
            events: dict[int, float] = defaultdict(float)
            for handle in handles:
                for start, end, value in handle.intervals(chromosome) or ():
                    if not math.isfinite(value):
                        raise ValueError("Combined BigWigs must contain only finite values.")
                    if value <= 0:
                        continue
                    events[int(start)] += float(value)
                    events[int(end)] -= float(value)
            current = 0.0
            previous = None
            for position in sorted(events):
                if previous is not None and position > previous and current > 0:
                    width = position - previous
                    weighted_sum += width * current
                    positive_bases += width
                current += events[position]
                if abs(current) < 1e-12:
                    current = 0.0
                previous = position
    finally:
        for handle in handles:
            handle.close()
    if positive_bases == 0:
        raise ValueError("Combined BigWigs contain no finite positive values.")
    return weighted_sum / positive_bases


def make_gene_only_config(
    config: dict[str, Any],
    *,
    head_id: str,
    correlation_loss_weight: float,
    row_correlation_loss_weight: float = 0.0,
    gene_supervision_path: str | None = None,
    output_rank: int | None = None,
    unstranded_output: bool = False,
) -> dict[str, Any]:
    """Copy a target manifest and retain one RNA head's gene supervision only."""
    if not math.isfinite(correlation_loss_weight) or correlation_loss_weight < 0:
        raise ValueError("Correlation loss weight must be finite and non-negative.")
    if not math.isfinite(row_correlation_loss_weight) or row_correlation_loss_weight < 0:
        raise ValueError("Row correlation loss weight must be finite and non-negative.")
    if output_rank is not None and (
        not isinstance(output_rank, int) or output_rank < 1
    ):
        raise ValueError("Output rank must be a positive integer.")
    config = copy.deepcopy(config)
    matches = [head for head in config.get("heads", ()) if head.get("id") == head_id]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one head named "{head_id}", found {len(matches)}.')
    head = matches[0]
    gene_supervision = head.get("gene_supervision")
    if not isinstance(gene_supervision, dict):
        raise ValueError(f'Head "{head_id}" does not define gene supervision.')
    gene_supervision["coverage_loss_weight"] = 0.0
    if gene_supervision_path is not None:
        gene_supervision["path"] = gene_supervision_path
    head["resolutions"] = [128]
    head["double_centered_correlation_loss_weight"] = correlation_loss_weight
    head["row_centered_correlation_loss_weight"] = row_correlation_loss_weight
    if unstranded_output:
        targets = head.get("targets")
        if not isinstance(targets, list) or len(targets) % 2:
            raise ValueError(
                "Unstranded gene output requires an interleaved +/- target pair per group."
            )
        collapsed_targets = []
        for positive, negative in zip(targets[0::2], targets[1::2], strict=True):
            if positive.get("strand") != "+" or negative.get("strand") != "-":
                raise ValueError(
                    "Unstranded gene output requires interleaved + then - targets."
                )
            target = copy.deepcopy(positive)
            target["strand"] = "."
            label = str(target.get("label", ""))
            if label.endswith(" (+)"):
                target["label"] = label[:-4]
            means = [
                float(item["nonzero_mean"])
                for item in (positive, negative)
                if item.get("nonzero_mean") is not None
            ]
            if means:
                target["nonzero_mean"] = combined_bigwig_nonzero_mean(
                    (Path(positive["path"]), Path(negative["path"]))
                )
            collapsed_targets.append(target)
        head["targets"] = collapsed_targets
    if output_rank is not None:
        if unstranded_output and output_rank >= len(head.get("targets", ())):
            raise ValueError("Output rank must be smaller than the RNA track count.")
        head["output_rank"] = output_rank
    return config


def set_head_output_rank(
    config: dict[str, Any],
    *,
    head_id: str,
    output_rank: int,
) -> dict[str, Any]:
    """Copy a target manifest and factorize one RNA head's output projection."""
    if not isinstance(output_rank, int) or output_rank < 1:
        raise ValueError("Output rank must be a positive integer.")
    config = copy.deepcopy(config)
    matches = [head for head in config.get("heads", ()) if head.get("id") == head_id]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one head named "{head_id}", found {len(matches)}.')
    head = matches[0]
    if str(head.get("kind", "")).lower() != "rna_seq":
        raise ValueError("Output factorization is currently supported only for RNA-seq heads.")
    if output_rank >= len(head.get("targets", ())):
        raise ValueError("Output rank must be smaller than the RNA track count.")
    head["output_rank"] = output_rank
    return config


def set_gene_window_assignment(
    config: dict[str, Any],
    *,
    head_id: str,
    assignment: str,
) -> dict[str, Any]:
    """Copy a target manifest and set one head's direct-gene window assignment."""
    if assignment not in {"full_span", "max_exon_overlap_scaled"}:
        raise ValueError("Gene window assignment must be full_span or max_exon_overlap_scaled.")
    config = copy.deepcopy(config)
    matches = [head for head in config.get("heads", ()) if head.get("id") == head_id]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one head named "{head_id}", found {len(matches)}.')
    gene_supervision = matches[0].get("gene_supervision")
    if not isinstance(gene_supervision, dict):
        raise ValueError(f'Head "{head_id}" does not define gene supervision.')
    gene_supervision["window_assignment"] = assignment
    return config


def retain_target_heads(
    config: dict[str, Any],
    head_ids: Iterable[str],
) -> dict[str, Any]:
    """Copy a target manifest and retain the requested heads in source order."""
    requested = tuple(dict.fromkeys(map(str, head_ids)))
    if not requested:
        raise ValueError("At least one target head must be retained.")
    config = copy.deepcopy(config)
    heads = config.get("heads")
    if not isinstance(heads, list):
        raise ValueError('Target manifest must contain a "heads" list.')
    available = {str(head.get("id")) for head in heads}
    missing = [head_id for head_id in requested if head_id not in available]
    if missing:
        raise ValueError(f"Target manifest does not contain heads {missing}.")
    requested_set = set(requested)
    config["heads"] = [
        head for head in heads if str(head.get("id")) in requested_set
    ]
    return config


def bigwig_nonzero_mean(path: Path) -> float:
    """Return the base-weighted mean over finite positive BigWig values."""
    weighted_sum = 0.0
    positive_bases = 0
    bigwig = pyBigWig.open(str(path))
    try:
        for chromosome in bigwig.chroms():
            intervals = bigwig.intervals(chromosome)
            if intervals is None:
                continue
            for start, end, value in intervals:
                if value > 0 and math.isfinite(value):
                    width = int(end) - int(start)
                    weighted_sum += width * float(value)
                    positive_bases += width
    finally:
        bigwig.close()
    if positive_bases == 0:
        raise ValueError(f"BigWig contains no finite positive values: {path}")
    return weighted_sum / positive_bases


def build_head_config(
    *,
    head_id: str,
    kind: str,
    tracks: Sequence[Path],
    labels: Sequence[str] | None = None,
    nonzero_means: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Construct one predefined-head configuration with stable track ordering."""
    tracks = tuple(Path(path).expanduser().resolve() for path in tracks)
    if not tracks:
        raise ValueError(f"Head {head_id!r} requires at least one target track.")
    if labels is None:
        labels = tuple(path.stem for path in tracks)
    if len(labels) != len(tracks):
        raise ValueError("labels must have one entry per target track.")
    if nonzero_means is not None and len(nonzero_means) != len(tracks):
        raise ValueError("nonzero_means must have one entry per target track.")

    targets = []
    for index, (path, label) in enumerate(zip(tracks, labels, strict=True)):
        if not path.exists():
            raise FileNotFoundError(f"Target BigWig does not exist: {path}")
        entry: dict[str, Any] = {"path": str(path), "label": str(label), "strand": "."}
        if nonzero_means is not None:
            mean = float(nonzero_means[index])
            if not math.isfinite(mean) or mean <= 0:
                raise ValueError(f"Invalid nonzero mean {mean} for {path}.")
            entry["nonzero_mean"] = mean
        targets.append(entry)

    return {
        "id": head_id,
        "source": "predefined",
        "kind": kind,
        "resolutions": [1, 128],
        "apply_squashing": kind == "rna_seq",
        "targets": targets,
    }


__all__ = [
    "bigwig_nonzero_mean",
    "build_head_config",
    "combined_bigwig_nonzero_mean",
    "make_gene_only_config",
    "retain_target_heads",
    "set_gene_window_assignment",
    "set_head_output_rank",
]
