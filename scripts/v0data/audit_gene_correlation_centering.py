#!/usr/bin/env python3
"""Audit minibatch versus split-wide centering for gene CPM objectives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune.data import _balance_gene_window_order


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
    chromosome_sizes: dict[str, int],
    include_chromosomes: set[str] | None,
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
    gene_keys = np.asarray(
        [
            f"{chromosome}:{start // window_size}"
            if chromosome not in excluded
            and end <= (start // window_size + 1) * window_size
            else ""
            for chromosome, start, end in zip(chromosomes, starts, ends, strict=True)
        ],
        dtype=object,
    )
    retained = gene_keys != ""
    values = values[retained]
    gene_keys = gene_keys[retained]
    if len(values) == 0:
        raise ValueError(f"No training genes remain in {path}.")

    row_centered = values - values.mean(axis=1, keepdims=True)
    globally_centered = row_centered - row_centered.mean(axis=0, keepdims=True)
    global_row_alignment = _cosine(row_centered, globally_centered)
    global_variance_fraction = float(
        np.sum(np.square(globally_centered)) / np.sum(np.square(row_centered))
    )

    order = np.argsort(gene_keys)
    sorted_keys = gene_keys[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1, len(order)]
    indices_by_window = {
        str(sorted_keys[boundaries[index]]): order[boundaries[index] : boundaries[index + 1]]
        for index in range(len(boundaries) - 1)
    }
    training_window_keys = [
        f"{chromosome}:{start // window_size}"
        for chromosome, size in chromosome_sizes.items()
        if chromosome not in excluded
        and (include_chromosomes is None or chromosome in include_chromosomes)
        and (
            include_chromosomes is not None
            or ("_" not in chromosome and not chromosome.startswith("chrUn"))
        )
        for start in range(0, size - window_size + 1, window_size)
    ]
    if not training_window_keys:
        raise ValueError(f"No training windows remain for {path}.")
    empty = np.empty((0,), dtype=np.int64)
    window_indices = [indices_by_window.get(key, empty) for key in training_window_keys]
    rng = np.random.default_rng(seed)
    local_alignments: list[float] = []
    genes_per_batch: list[float] = []
    batches_without_local_variance = 0
    balanced_local_alignments: list[float] = []
    balanced_genes_per_batch: list[float] = []
    balanced_batches_without_local_variance = 0

    def record_batches(
        permutation: np.ndarray,
        batch_genes: list[float],
        alignments: list[float],
    ) -> int:
        without_local_variance = 0
        usable = len(permutation) // batch_size * batch_size
        for begin in range(0, usable, batch_size):
            groups = [window_indices[index] for index in permutation[begin : begin + batch_size]]
            nonempty = [indices for indices in groups if len(indices)]
            indices = np.concatenate(nonempty) if nonempty else empty
            batch_genes.append(float(len(indices)))
            if len(indices) == 0:
                continue
            batch = values[indices]
            local = batch - batch.mean(axis=1, keepdims=True)
            local = local - local.mean(axis=0, keepdims=True)
            alignment = _cosine(local, globally_centered[indices])
            if np.isfinite(alignment):
                alignments.append(alignment)
            else:
                without_local_variance += 1
        return without_local_variance

    window_gene_counts = np.asarray([len(indices) for indices in window_indices], dtype=np.int64)
    for _ in range(repeats):
        permutation = rng.permutation(len(window_indices))
        batches_without_local_variance += record_batches(
            permutation, genes_per_batch, local_alignments
        )
        balanced = _balance_gene_window_order(
            permutation,
            window_gene_counts,
            batch_size=batch_size,
            drop_last=True,
        )
        balanced_batches_without_local_variance += record_batches(
            balanced, balanced_genes_per_batch, balanced_local_alignments
        )

    return {
        "gene_supervision": str(path),
        "groups": int(values.shape[1]),
        "training_genes": int(values.shape[0]),
        "training_windows": len(window_indices),
        "training_windows_with_genes": int(sum(len(indices) > 0 for indices in window_indices)),
        "fraction_training_windows_with_genes": float(
            np.mean([len(indices) > 0 for indices in window_indices])
        ),
        "batch_size_windows": batch_size,
        "shuffle_repeats": repeats,
        "row_vs_global_double_centered_cosine": global_row_alignment,
        "global_double_centered_variance_fraction_of_row_centered": global_variance_fraction,
        "local_vs_global_double_centered_cosine": _summary(local_alignments),
        "genes_per_batch": _summary(genes_per_batch),
        "fraction_batches_without_genes": float(np.mean(np.asarray(genes_per_batch) == 0)),
        "fraction_batches_without_local_double_centered_variance": float(
            batches_without_local_variance / len(genes_per_batch)
        ),
        "balanced_local_vs_global_double_centered_cosine": _summary(
            balanced_local_alignments
        ),
        "balanced_genes_per_batch": _summary(balanced_genes_per_batch),
        "balanced_fraction_batches_without_genes": float(
            np.mean(np.asarray(balanced_genes_per_batch) == 0)
        ),
        "balanced_fraction_batches_without_local_double_centered_variance": float(
            balanced_batches_without_local_variance / len(balanced_genes_per_batch)
        ),
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


def _parse_chromosomes(value: str) -> tuple[str, set[str]]:
    label, separator, chromosomes = value.partition("=")
    if not separator or not label or not chromosomes:
        raise argparse.ArgumentTypeError("Chromosome sets must use LABEL=CHR1;CHR2 syntax.")
    return label, set(filter(None, chromosomes.split(";")))


def _parse_labeled_positive_int(value: str) -> tuple[str, int]:
    label, separator, raw_number = value.partition("=")
    if not separator or not label or not raw_number:
        raise argparse.ArgumentTypeError("Values must use LABEL=INTEGER syntax.")
    try:
        number = int(raw_number)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Values must use LABEL=INTEGER syntax.") from error
    if number < 1:
        raise argparse.ArgumentTypeError("The integer must be positive.")
    return label, number


def _read_fai(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                sizes[fields[0]] = int(fields[1])
    if not sizes:
        raise ValueError(f"No chromosome sizes found in {path}.")
    return sizes


def _render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Gene-correlation centering audit",
        "",
        "Gene targets are counts per million, CPM. `Row/global cosine` compares targets centered only across cell groups within each gene with targets also centered across all training genes. `Local/global cosine` compares minibatch double centering with split-wide double centering after randomizing genomic windows. A low local value indicates noise from estimating cell-group means using the few genes in one sequence batch.",
        "",
        "| Dataset | Groups | Training genes | Gene-bearing windows | Random median genes | Balanced median genes | Random empty | Balanced empty | Random no local DC variance | Balanced no local DC variance | Row/global cosine | Global DC variance retained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, audit in result["datasets"].items():
        genes = audit["genes_per_batch"]
        balanced_genes = audit["balanced_genes_per_batch"]
        lines.append(
            f"| {label} | {audit['groups']} | {audit['training_genes']:,} | "
            f"{audit['fraction_training_windows_with_genes']:.1%} | {genes['median']:.0f} | "
            f"{balanced_genes['median']:.0f} | "
            f"{audit['fraction_batches_without_genes']:.1%} | "
            f"{audit['balanced_fraction_batches_without_genes']:.1%} | "
            f"{audit['fraction_batches_without_local_double_centered_variance']:.1%} | "
            f"{audit['balanced_fraction_batches_without_local_double_centered_variance']:.1%} | "
            f"{audit['row_vs_global_double_centered_cosine']:.5f} | "
            f"{audit['global_double_centered_variance_fraction_of_row_centered']:.5f} |"
        )
    lines.extend(
        [
            "",
            "For CPM targets, total expression is nearly equal across cell-group tracks. Row-centering therefore closely approximates split-wide double centering without estimating a noisy cell-group mean from each small minibatch. Gene-balanced ordering is intended only for the row-centered objective. In sparse Johansen data it reduces empty batches but increases batches containing only one gene, which have no local double-centered variance. This is an objective-alignment diagnostic, not a prediction result.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--fasta-index", action="append", type=_parse_dataset, required=True)
    parser.add_argument(
        "--held-out",
        action="append",
        type=_parse_held_out,
        default=[],
        help="Optional per-dataset chromosome sets as LABEL=VALID,TEST; separate sets with semicolons.",
    )
    parser.add_argument(
        "--include-chromosomes",
        action="append",
        type=_parse_chromosomes,
        default=[],
        help="Optional native chromosome whitelist as LABEL=CHR1;CHR2.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--dataset-batch-size",
        action="append",
        default=[],
        type=_parse_labeled_positive_int,
        help="Override the global window batch size as LABEL=SIZE.",
    )
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
    held_out = {
        label: (dataset_valid, dataset_test)
        for label, dataset_valid, dataset_test in args.held_out
    }
    fasta_indices = dict(args.fasta_index)
    included = dict(args.include_chromosomes)
    dataset_batch_sizes = dict(args.dataset_batch_size)
    dataset_labels = {label for label, _ in args.dataset}
    unknown = (
        set(held_out) | set(fasta_indices) | set(included) | set(dataset_batch_sizes)
    ) - dataset_labels
    if unknown:
        raise ValueError(f"Dataset options reference unknown datasets: {sorted(unknown)}")
    missing_fasta = dataset_labels - set(fasta_indices)
    if missing_fasta:
        raise ValueError(f"Missing FASTA indexes for datasets: {sorted(missing_fasta)}")
    result = {
        "datasets": {
            label: audit_supervision(
                path.expanduser(),
                chromosome_sizes=_read_fai(fasta_indices[label].expanduser()),
                include_chromosomes=included.get(label),
                batch_size=dataset_batch_sizes.get(label, args.batch_size),
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
