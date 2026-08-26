#!/usr/bin/env python3
"""Audit the chromosome-9 RNA outlier in Zemke 2023 marmoset targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyBigWig

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v0data.zemke2023_rna_reprocessing.prepare_marmoset_ano6_audit import (
    OUTLIER_CHROMOSOME,
    OUTLIER_END,
    OUTLIER_LABEL,
    OUTLIER_START,
)


def _rna_head(path: Path) -> dict[str, Any]:
    heads = json.loads(path.read_text())["heads"]
    matches = [head for head in heads if head.get("kind") == "rna_seq"]
    if len(matches) != 1:
        raise ValueError(f"Expected one RNA head in {path}, found {len(matches)}.")
    return matches[0]


def _coarse_chromosome_matrix(
    head: dict[str, Any], chromosome: str, num_bins: int
) -> tuple[np.ndarray, int]:
    columns = []
    chromosome_size = None
    for target in head["targets"]:
        with pyBigWig.open(str(target["path"])) as bigwig:
            current_size = bigwig.chroms().get(chromosome)
            if current_size is None:
                raise ValueError(f"{target['path']} lacks {chromosome}.")
            if chromosome_size is None:
                chromosome_size = current_size
            elif current_size != chromosome_size:
                raise ValueError("RNA tracks have inconsistent chromosome sizes.")
            values = bigwig.stats(
                chromosome,
                0,
                current_size,
                nBins=num_bins,
                type="mean",
                exact=True,
            )
        columns.append(np.asarray([0.0 if value is None else value for value in values]))
    return np.stack(columns, axis=1), int(chromosome_size)


def target_leverage(
    values: np.ndarray,
    *,
    chromosome_size: int,
    excluded_start: int | None = None,
    excluded_end: int | None = None,
) -> dict[str, Any]:
    """Summarize double-centered variance across coarse loci and tracks."""
    centered = values - values.mean(axis=0) - values.mean(axis=1, keepdims=True) + values.mean()
    locus_ss = np.sum(np.square(centered), axis=1)
    track_ss = np.sum(np.square(centered), axis=0)
    total_ss = float(locus_ss.sum())
    result = {
        "nonzero_fraction": float(np.mean(values > 0)),
        "standard_deviation": float(values.std()),
        "double_centered_standard_deviation": float(centered.std()),
        "median_pairwise_track_correlation": float(
            np.nanmedian(np.corrcoef(values, rowvar=False)[np.triu_indices(values.shape[1], 1)])
        ),
        "maximum_locus_variance_fraction": float(locus_ss.max() / total_ss),
        "maximum_track_variance_fraction": float(track_ss.max() / total_ss),
        "maximum_track_index": int(track_ss.argmax()),
    }
    if excluded_start is not None and excluded_end is not None:
        starts = np.floor(
            np.arange(values.shape[0], dtype=np.float64) * chromosome_size / values.shape[0]
        ).astype(np.int64)
        ends = np.ceil(
            np.arange(1, values.shape[0] + 1, dtype=np.float64)
            * chromosome_size
            / values.shape[0]
        ).astype(np.int64)
        excluded = (starts < excluded_end) & (excluded_start < ends)
        retained = values[~excluded]
        retained_centered = (
            retained
            - retained.mean(axis=0)
            - retained.mean(axis=1, keepdims=True)
            + retained.mean()
        )
        result.update(
            excluded_bin_count=int(excluded.sum()),
            excluded_locus_variance_fraction=float(locus_ss[excluded].sum() / total_ss),
            retained_double_centered_standard_deviation=float(retained_centered.std()),
        )
    return result


def _gene_summary(supervision_path: Path, gene_id: str) -> dict[str, Any]:
    with np.load(supervision_path, allow_pickle=False) as supervision:
        ids = supervision["gene_ids"].astype(str)
        matches = np.flatnonzero(ids == gene_id)
        if len(matches) != 1:
            return {"present": False}
        index = int(matches[0])
        cpm = np.asarray(supervision["cpm"][:, index], dtype=np.float64)
        groups = supervision["groups"].astype(str)
        return {
            "present": True,
            "chromosome": str(supervision["chromosomes"][index]),
            "start": int(supervision["starts"][index]),
            "end": int(supervision["ends"][index]),
            "maximum_cpm": float(cpm.max()),
            "maximum_cpm_group": str(groups[int(cpm.argmax())]),
            "median_cpm": float(np.median(cpm)),
        }


def _outlier_region_summary(head: dict[str, Any]) -> dict[str, Any]:
    track_maxima = {}
    chromosome_maxima = {}
    for index, target in enumerate(head["targets"]):
        label = str(target.get("label", index))
        with pyBigWig.open(str(target["path"])) as bigwig:
            maximum = bigwig.stats(
                OUTLIER_CHROMOSOME,
                OUTLIER_START,
                OUTLIER_END,
                type="max",
                exact=True,
            )[0]
            chromosome_maximum = bigwig.stats(
                OUTLIER_CHROMOSOME,
                type="max",
                exact=True,
            )[0]
        track_maxima[label] = 0.0 if maximum is None else float(maximum)
        chromosome_maxima[label] = (
            0.0 if chromosome_maximum is None else float(chromosome_maximum)
        )
    values = np.asarray(tuple(track_maxima.values()), dtype=np.float64)
    return {
        "label": OUTLIER_LABEL,
        "chromosome": OUTLIER_CHROMOSOME,
        "start": OUTLIER_START,
        "end": OUTLIER_END,
        "track_maxima": track_maxima,
        "chromosome_maxima": chromosome_maxima,
        "tracks_with_chromosome_maximum_in_region": int(
            sum(
                bool(np.isclose(track_maxima[label], chromosome_maxima[label]))
                for label in track_maxima
            )
        ),
        "minimum_track_maximum": float(values.min()),
        "median_track_maximum": float(np.median(values)),
        "maximum_track_maximum": float(values.max()),
        "maximum_track_label": max(track_maxima, key=track_maxima.get),
    }


def _matched_evaluation(
    baseline_path: Path, excluded_path: Path
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text())
    excluded = json.loads(excluded_path.read_text())
    for key in ("source_epoch", "source_global_step"):
        if baseline.get(key) != excluded.get(key):
            raise ValueError(f"Matched evaluations differ in {key}.")
    rows = {}
    for head in ("zemke2023_atac", "zemke2023_rna"):
        baseline_r = float(baseline["metrics"]["test"][head]["differential_pearson_r"])
        excluded_r = float(excluded["metrics"]["test"][head]["differential_pearson_r"])
        rows[head] = {
            "baseline_r": baseline_r,
            "repeat_excluded_r": excluded_r,
            "difference": excluded_r - baseline_r,
        }
    return {
        "source_epoch": int(baseline["source_epoch"]),
        "source_global_step": int(baseline["source_global_step"]),
        "baseline_evaluation": str(baseline_path),
        "repeat_excluded_evaluation": str(excluded_path),
        "baseline_windows": 1022,
        "repeat_excluded_windows": 1021,
        "heads": rows,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Zemke 2023 marmoset RNA test outlier audit",
        "",
        "RNA target tracks are summarized in equal-width genomic bins. Double-centered standard deviation (DC SD) and variance fractions refer to the target after centering over genomic bins and cell-subclass tracks.",
        "",
        "| Species | Chromosome | Nonzero | DC SD | Median track R | Max locus variance | Max track variance |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["chromosome_audits"]:
        lines.append(
            f"| {row['species']} | {row['chromosome']} | {row['nonzero_fraction']:.4f} | "
            f"{row['double_centered_standard_deviation']:.4f} | "
            f"{row['median_pairwise_track_correlation']:.4f} | "
            f"{row['maximum_locus_variance_fraction']:.4f} | "
            f"{row['maximum_track_variance_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## ANO6 expression",
            "",
            "| Species | Chromosome | Max CPM | Max group | Median CPM |",
            "|---|---|---:|---|---:|",
        ]
    )
    for species, row in result["gene_summaries"].items():
        if not row["present"]:
            lines.append(f"| {species} | not matched | NA | NA | NA |")
        else:
            lines.append(
                f"| {species} | {row['chromosome']} | {row['maximum_cpm']:.2f} | "
                f"{row['maximum_cpm_group']} | {row['median_cpm']:.2f} |"
            )
    marmoset_test = next(
        row
        for row in result["chromosome_audits"]
        if row["species"] == "marmoset" and row["chromosome"] == "chr9"
    )
    lines.extend(
        [
            "",
            f"The ribosomal repeat contains the chromosome-wide maximum in {result['outlier_region']['tracks_with_chromosome_maximum_in_region']} of 20 marmoset RNA tracks. Repeat-region maxima range from {result['outlier_region']['minimum_track_maximum']:.1f} to {result['outlier_region']['maximum_track_maximum']:.1f} RPKM, with the largest value in {result['outlier_region']['maximum_track_label']}.",
            "",
            f"The coarse bin overlapping the {OUTLIER_LABEL} repeat inside marmoset ANO6 accounts for "
            f"{marmoset_test['excluded_locus_variance_fraction']:.1%} of chromosome-9 "
            "double-centered target variance.",
            "",
        ]
    )
    if "matched_evaluation" in result:
        lines.extend(
            [
                "## Matched checkpoint evaluation",
                "",
                "| Target | Baseline R | Repeat-window excluded R | Difference |",
                "|---|---:|---:|---:|",
            ]
        )
        for head, row in result["matched_evaluation"]["heads"].items():
            lines.append(
                f"| {head.removeprefix('zemke2023_').upper()} | {row['baseline_r']:.4f} | "
                f"{row['repeat_excluded_r']:.4f} | {row['difference']:+.4f} |"
            )
        lines.extend(
            [
                "",
                "Both evaluations restore the same selected checkpoint. The baseline uses all 1,022 chromosome-9 windows; the diagnostic omits only the one 131 kb window overlapping the ribosomal repeat.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-root", required=True, type=Path)
    parser.add_argument("--supervision-root", required=True, type=Path)
    parser.add_argument("--num-bins", type=int, default=2048)
    parser.add_argument("--gene-id", default="ANO6")
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    parser.add_argument("--baseline-evaluation", type=Path)
    parser.add_argument("--excluded-evaluation", type=Path)
    args = parser.parse_args()
    species_names = ("human", "macaque", "marmoset", "mouse")
    gene_summaries = {
        species: _gene_summary(
            args.supervision_root / species / "gene_expression_supervision.npz",
            args.gene_id,
        )
        for species in species_names
    }
    marmoset_gene = gene_summaries["marmoset"]
    if not marmoset_gene["present"]:
        raise ValueError(f"Marmoset supervision lacks {args.gene_id}.")
    marmoset_head = _rna_head(args.species_root / "marmoset" / "targets.json")
    chromosome_audits = []
    for species in species_names:
        head = _rna_head(args.species_root / species / "targets.json")
        for chromosome in ("chr8", "chr9"):
            values, chromosome_size = _coarse_chromosome_matrix(
                head, chromosome, args.num_bins
            )
            exclusions = (
                {
                    "excluded_start": OUTLIER_START,
                    "excluded_end": OUTLIER_END,
                }
                if species == "marmoset" and chromosome == "chr9"
                else {}
            )
            summary = target_leverage(
                values,
                chromosome_size=chromosome_size,
                **exclusions,
            )
            summary.update(
                species=species,
                chromosome=chromosome,
                chromosome_size=chromosome_size,
                maximum_track_label=str(
                    head["targets"][summary["maximum_track_index"]].get(
                        "label", summary["maximum_track_index"]
                    )
                ),
            )
            chromosome_audits.append(summary)
    result = {
        "gene_id": args.gene_id,
        "excluded_repeat": {
            "label": OUTLIER_LABEL,
            "chromosome": OUTLIER_CHROMOSOME,
            "start": OUTLIER_START,
            "end": OUTLIER_END,
            "repeat_class": "rRNA",
            "repeat_family": "rRNA",
        },
        "num_equal_width_bins": args.num_bins,
        "gene_summaries": gene_summaries,
        "outlier_region": _outlier_region_summary(marmoset_head),
        "chromosome_audits": chromosome_audits,
    }
    if (args.baseline_evaluation is None) != (args.excluded_evaluation is None):
        parser.error("--baseline-evaluation and --excluded-evaluation must be provided together")
    if args.baseline_evaluation is not None:
        result["matched_evaluation"] = _matched_evaluation(
            args.baseline_evaluation,
            args.excluded_evaluation,
        )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(result))


if __name__ == "__main__":
    main()
