"""Backend interface for unified finetuning entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence


BackendName = Literal["jax", "torch"]


@dataclass(frozen=True)
class PreparedRun:
    """Data prepared by the shared pipeline before backend-specific training."""

    bigwig_dir: Path
    bigwigs: Sequence[Path]
    fasta_path: Path
    intervals: Mapping[str, Sequence[object]]
    head_specs: Sequence[Any]


class FinetuneBackend(Protocol):
    """Common backend contract for finetuning launchers."""

    name: BackendName

    def run(self, prepared: PreparedRun) -> None:
        """Run backend-specific training."""


__all__ = ["BackendName", "PreparedRun", "FinetuneBackend"]
