#!/usr/bin/env python3
"""Audit minibatch versus split-wide centering for gene CPM objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.vdot(first, second) / denominator) if denominator > 0 else float("nan")


def audit_supervision(
    path: Path,
    *,
    batch_size: int,
    repeats: int,
    seed: int,
    valid_chromosomes: set[str],
    test_chromosomes: set[str],
    window_size: int,
) -> dict[str, object]:
    """Compare local double centering with row and split-wide centering."""
    if batch_size < 1 or repeats < 1:
        raise ValueError("batch_size and repeats must be positive.")
    with np.load(path, allow_pickle=False) as data:
        values = np.asarray(data["cpm"], dtype=np.float64).T
        chromosomes = np.asarray(data["chromosomes"]).astype(str)
        starts = np.asarray(data["starts"], dtype=np.int64)
        ends = np.asarray(data["ends"], dtype=np.int64)

    excluded = valid_chromosomes | test_chromosomes | {"chrM", "chrY"}
    keys = np.asarray(
        [
            f"{chromosome}:{start // window_size}"
            if chromosome not in excluded
            and end <= (start // window_size + 1) * window_size
            else ""
            for chromosome, start, end in zip(chromosomes, starts, ends, strict=True)
        ],
        dtype=object,
    )
    retained = keys != ""
    values = values[retained]
    keys = keys[retained]
    if len(values) == 0:
        raise ValueError(f"No training genes remain in {path}.")

    row_centered = values - values.mean(axis=1, keepdims=True)
    globally_centered = row_centered - row_centered.mean(axis=0, keepdims=True)
    global_row_alignment = _cosine(row_centered, globally_centered)
    global_variance_fraction = float(
        np.sum(np.square(globally_centered)) / np.sum(np.square(row_centered))
    )

    order = np.argsort(keys)
    sorted_keys = keys[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1, len(order)]
    window_indices = [
        order[boundaries[index] : boundaries[index + 1]]
        for index in range(len(boundaries) - 1)
    ]
    rng = np.random.default_rng(seed)
    local_alignments: list[float] = []
    genes_per_batch: list[float] = []
    for _ in range(repeats):
        permutation = rng.permutation(len(window_indices))
        usable = len(permutation) // batch_size * batch_size
        for begin in range(0, usable, batch_size):
            indices = np.concatenate(
                [window_indices[index] for index in permutation[begin : begin + batch_size]]
            )
            batch = values[indices]
            local = batch - batch.mean(axis=1, keepdims=True)
            local = local - local.mean(axis=0, keepdims=True)
            local_alignments.append(_cosine(local, globally_centered[indices]))
            genes_per_batch.append(float(len(indices)))

    return {
        "gene_supervision": str(path),
        "groups": int(values.shape[1]),
        "training_genes": int(values.shape[0]),
        "training_windows_with_genes": len(window_indices),
        "batch_size_windows": batch_size,
        "shuffle_repeats": repeats,
        "row_vs_global_double_centered_cosine": global_row_alignment,
        "global_double_centered_variance_fraction_of_row_centered": global_variance_fraction,
        "local_vs_global_double_centered_cosine": _summary(local_alignments),
        "genes_per_batch": _summary(genes_per_batch),
    }


def _parse_dataset(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("Datasets must use LABEL=PATH syntax.")
    return label, Path(path)


def _parse_held_out(value: str) -> tuple[str, set[str], set[str]]:
    label, separator, chromosomes = value.partition("=")
    valid, second_separator, test = chromosomes.partition(",")
    if not separator or not second_separator or not label or not valid or not test:
        raise argparse.ArgumentTypeError("Held-out sets must use LABEL=VALID,TEST syntax.")
    return label, set(filter(None, valid.split(";"))), set(filter(None, test.split(";")))


def _render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Gene-correlation centering audit",
        "",
        "Gene targets are counts per million, CPM. `Row/global cosine` compares targets centered only across cell groups within each gene with targets also centered across all training genes. `Local/global cosine` compares minibatch double centering with split-wide double centering after randomizing genomic windows. A low local value indicates noise from estimating cell-group means using the few genes in one sequence batch.",
        "",
        "| Dataset | Groups | Training genes | Median genes/batch | Row/global cosine | Global DC variance retained | Median local/global cosine | Local/global p10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, audit in result["datasets"].items():
        local = audit["local_vs_global_double_centered_cosine"]
        genes = audit["genes_per_batch"]
        lines.append(
            f"| {label} | {audit['groups']} | {audit['training_genes']:,} | "
            f"{genes['median']:.0f} | {audit['row_vs_global_double_centered_cosine']:.5f} | "
            f"{audit['global_double_centered_variance_fraction_of_row_centered']:.5f} | "
            f"{local['median']:.5f} | {local['p10']:.5f} |"
        )
    lines.extend(
        [
            "",
            "For CPM targets, total expression is nearly equal across cell-group tracks. Row-centering therefore closely approximates split-wide double centering without estimating a noisy cell-group mean from each small minibatch. This is an objective-alignment diagnostic, not a prediction result.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument(
        "--held-out",
        action="append",
        type=_parse_held_out,
        default=[],
        help="Optional per-dataset chromosome sets as LABEL=VALID,TEST; separate sets with semicolons.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--window-size", type=int, default=131072)
    parser.add_argument("--valid-chromosomes", default="chr8")
    parser.add_argument("--test-chromosomes", default="chr9")
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()

    valid = set(filter(None, args.valid_chromosomes.split(",")))
    test = set(filter(None, args.test_chromosomes.split(",")))
    held_out = {label: (dataset_valid, dataset_test) for label, dataset_valid, dataset_test in args.held_out}
    unknown = set(held_out) - {label for label, _ in args.dataset}
    if unknown:
        raise ValueError(f"Held-out chromosomes reference unknown datasets: {sorted(unknown)}")
    result = {
        "datasets": {
            label: audit_supervision(
                path.expanduser().resolve(),
                batch_size=args.batch_size,
                repeats=args.repeats,
                seed=args.seed,
                valid_chromosomes=held_out.get(label, (valid, test))[0],
                test_chromosomes=held_out.get(label, (valid, test))[1],
                window_size=args.window_size,
            )
            for label, path in args.dataset
        }
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(_render_markdown(result))
    print(_render_markdown(result), end="")


if __name__ == "__main__":
    main()
