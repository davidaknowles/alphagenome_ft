#!/usr/bin/env python3
"""Report sequence composition for selected FASTA chromosomes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from pyfaidx import Fasta


def audit_chromosome(reference: Fasta, chromosome: str, chunk_size: int) -> dict[str, object]:
    if chromosome not in reference:
        raise ValueError(f"Chromosome {chromosome!r} is absent from the FASTA.")
    length = len(reference[chromosome])
    counts: Counter[str] = Counter()
    lowercase_bases = 0
    for start in range(0, length, chunk_size):
        sequence = str(reference[chromosome][start : min(start + chunk_size, length)])
        lowercase_bases += sum(base.islower() for base in sequence)
        counts.update(sequence.upper())

    canonical_bases = sum(counts[base] for base in "ACGT")
    gc_bases = counts["G"] + counts["C"]
    n_bases = counts["N"]
    return {
        "chromosome": chromosome,
        "length": length,
        "base_counts": dict(sorted(counts.items())),
        "canonical_fraction": canonical_bases / length,
        "gc_fraction_of_canonical": gc_bases / canonical_bases if canonical_bases else None,
        "n_fraction": n_bases / length,
        "softmasked_fraction": lowercase_bases / length,
        "unexpected_fraction": (length - canonical_bases - n_bases) / length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("chromosomes", nargs="+")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive.")

    reference = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=False)
    try:
        results = {
            "fasta": str(args.fasta),
            "chromosomes": [
                audit_chromosome(reference, chromosome, args.chunk_size)
                for chromosome in args.chromosomes
            ],
        }
    finally:
        reference.close()

    rendered = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
