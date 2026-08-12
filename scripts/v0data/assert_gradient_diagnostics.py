#!/usr/bin/env python3
"""Require finite, nonzero head gradients from technical smoke runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def assert_head_gradients(
    paths: list[Path],
    *,
    expected_heads: set[str],
    minimum_norm: float,
) -> dict[str, dict[str, float]]:
    """Validate and return head-gradient norms grouped by run path."""
    if minimum_norm < 0 or not math.isfinite(minimum_norm):
        raise ValueError("minimum_norm must be finite and non-negative.")
    results: dict[str, dict[str, float]] = {}
    for path in paths:
        payload: dict[str, Any] = json.loads(path.read_text())
        heads = payload.get("heads")
        if not isinstance(heads, dict):
            raise ValueError(f"Missing head diagnostics in {path}.")
        missing = expected_heads - set(heads)
        if missing:
            raise ValueError(f"Missing heads in {path}: {sorted(missing)}")
        norms = {
            head: float(heads[head]["head_gradient_norm"])
            for head in sorted(expected_heads)
        }
        invalid = {
            head: norm
            for head, norm in norms.items()
            if not math.isfinite(norm) or norm <= minimum_norm
        }
        if invalid:
            raise ValueError(
                f"Head-gradient gate failed for {path}: {invalid}; "
                f"required norms greater than {minimum_norm}."
            )
        results[str(path)] = norms
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-head", action="append", required=True)
    parser.add_argument("--minimum-norm", type=float, default=0.0)
    args = parser.parse_args()

    results = assert_head_gradients(
        args.paths,
        expected_heads=set(args.expected_head),
        minimum_norm=args.minimum_norm,
    )
    for path, norms in results.items():
        print(path, ", ".join(f"{head}={norm:.6g}" for head, norm in norms.items()))


if __name__ == "__main__":
    main()
