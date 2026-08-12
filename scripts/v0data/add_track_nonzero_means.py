#!/usr/bin/env python3
"""Add exact per-track nonzero means to selected target-manifest heads."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import bigwig_nonzero_mean


def add_track_nonzero_means(
    config: Mapping[str, Any],
    head_ids: Sequence[str],
    *,
    workers: int = 1,
    mean_fn: Callable[[Path], float] = bigwig_nonzero_mean,
) -> dict[str, Any]:
    """Return a copied manifest with nonzero means populated for selected heads."""
    if workers < 1:
        raise ValueError("workers must be positive.")
    requested = set(head_ids)
    if not requested:
        raise ValueError("At least one head ID is required.")

    result = copy.deepcopy(dict(config))
    heads = result.get("heads")
    if not isinstance(heads, list):
        raise ValueError('Target manifest must contain a "heads" list.')
    available = {str(head.get("id")) for head in heads if isinstance(head, Mapping)}
    missing = requested - available
    if missing:
        raise ValueError(f"Unknown head IDs: {sorted(missing)}")

    selected_targets: list[dict[str, Any]] = []
    selected_paths: list[Path] = []
    for head in heads:
        if str(head.get("id")) not in requested:
            continue
        targets = head.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f'Head {head.get("id")!r} has no target tracks.')
        for target in targets:
            if not isinstance(target, dict) or "path" not in target:
                raise ValueError(f'Head {head.get("id")!r} has an invalid target entry.')
            selected_targets.append(target)
            selected_paths.append(Path(str(target["path"])).expanduser())

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        means = list(executor.map(mean_fn, selected_paths))
    for target, mean in zip(selected_targets, means, strict=True):
        if not 0 < mean < float("inf"):
            raise ValueError(f"Invalid nonzero mean {mean!r} for {target['path']}")
        target["nonzero_mean"] = float(mean)

    contract = result.setdefault("target_contract", {})
    if not isinstance(contract, dict):
        raise ValueError('Target manifest "target_contract" must be an object when present.')
    contract["nonzero_mean_scaled_heads"] = sorted(requested)
    contract["nonzero_mean_definition"] = "base-weighted finite positive values"
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heads", required=True, help="Comma-separated head IDs.")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = json.loads(args.input.expanduser().read_text())
    head_ids = [value.strip() for value in args.heads.split(",") if value.strip()]
    result = add_track_nonzero_means(config, head_ids, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    counts = {
        head["id"]: len(head["targets"])
        for head in result["heads"]
        if head["id"] in set(head_ids)
    }
    print(f"Wrote nonzero means for {counts} to {args.output}")


if __name__ == "__main__":
    main()
