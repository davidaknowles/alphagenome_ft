"""Utilities for constructing validated BigWig target manifests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import pyBigWig


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


__all__ = ["bigwig_nonzero_mean", "build_head_config"]
