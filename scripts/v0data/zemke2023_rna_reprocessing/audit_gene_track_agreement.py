#!/usr/bin/env python3
"""Compare raw-count CPM with published RPKM integrated over union exons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyBigWig


def double_centered_r(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation after centering both matrix dimensions."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Inputs must be two matrices with the same shape.")
    left = left - left.mean(axis=0, keepdims=True) - left.mean(axis=1, keepdims=True) + left.mean()
    right = right - right.mean(axis=0, keepdims=True) - right.mean(axis=1, keepdims=True) + right.mean()
    left = left.ravel()
    right = right.ravel()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator == 0:
        raise ValueError("Double-centered matrices must have nonzero variance.")
    return float((left @ right) / denominator)


def row_pattern_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return per-gene cell-group correlations after removing gene-specific scale."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Inputs must be two matrices with the same shape.")
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominator > 0
    if not np.any(valid):
        raise ValueError("No rows have nonzero variance in both matrices.")
    return np.sum(left[valid] * right[valid], axis=1) / denominator[valid]


def _integrate_track(
    path: Path,
    *,
    chromosomes: np.ndarray,
    exon_offsets: np.ndarray,
    exon_starts: np.ndarray,
    exon_ends: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    values = np.zeros(len(indices), dtype=np.float64)
    with pyBigWig.open(str(path)) as bigwig:
        available = bigwig.chroms()
        for output_index, gene_index in enumerate(indices):
            chromosome = str(chromosomes[gene_index])
            if chromosome not in available:
                continue
            start_offset = int(exon_offsets[gene_index])
            end_offset = int(exon_offsets[gene_index + 1])
            total = 0.0
            for start, end in zip(
                exon_starts[start_offset:end_offset],
                exon_ends[start_offset:end_offset],
                strict=True,
            ):
                result = bigwig.stats(
                    chromosome,
                    int(start),
                    int(end),
                    type="sum",
                    exact=True,
                )[0]
                if result is not None and math.isfinite(result):
                    total += float(result)
            values[output_index] = total / 1000.0
    return values


def audit_species(
    species: str,
    species_dir: Path,
    *,
    sample_genes: int,
    seed: int,
) -> dict[str, Any]:
    manifest = json.loads((species_dir / "manifest.json").read_text())
    targets = json.loads(Path(manifest["targets"]).read_text())
    rna_heads = [head for head in targets["heads"] if head.get("kind") == "rna_seq"]
    if len(rna_heads) != 1:
        raise ValueError(f"{species} must contain one RNA head.")
    rna_head = rna_heads[0]
    target_groups = tuple(str(target["label"]) for target in rna_head["targets"])
    with np.load(manifest["gene_supervision"]) as supervision:
        groups = tuple(str(value) for value in supervision["groups"])
        if groups != target_groups:
            raise ValueError(f"{species} gene groups do not match RNA target order.")
        cpm = np.asarray(supervision["cpm"], dtype=np.float64)
        group_valid = (
            np.asarray(supervision["group_valid"], dtype=bool)
            if "group_valid" in supervision
            else np.ones((len(groups),), dtype=bool)
        )
        chromosomes = supervision["chromosomes"].astype(str)
        exon_offsets = np.asarray(supervision["exon_offsets"], dtype=np.int64)
        exon_starts = np.asarray(supervision["exon_starts"], dtype=np.int64)
        exon_ends = np.asarray(supervision["exon_ends"], dtype=np.int64)
    if cpm.shape[0] != len(groups) or cpm.shape[1] != len(chromosomes):
        raise ValueError(f"{species} supervision arrays have inconsistent dimensions.")
    if group_valid.shape != (len(groups),) or not np.any(group_valid):
        raise ValueError(f"{species} has invalid direct-gene group validity.")
    eligible = np.flatnonzero(np.diff(exon_offsets) > 0)
    if not len(eligible):
        raise ValueError(f"{species} has no annotated genes.")
    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(eligible, size=min(sample_genes, len(eligible)), replace=False)
    )
    integrated = np.stack(
        [
            _integrate_track(
                Path(target["path"]),
                chromosomes=chromosomes,
                exon_offsets=exon_offsets,
                exon_starts=exon_starts,
                exon_ends=exon_ends,
                indices=indices,
            )
            for target, valid in zip(rna_head["targets"], group_valid, strict=True)
            if valid
        ],
        axis=1,
    )
    direct = cpm[group_valid][:, indices].T
    raw_patterns = row_pattern_correlations(direct, integrated)
    log_patterns = row_pattern_correlations(np.log1p(direct), np.log1p(integrated))
    return {
        "species": species,
        "sampled_genes": len(indices),
        "groups": int(np.sum(group_valid)),
        "unsupported_groups": [
            group for group, valid in zip(groups, group_valid, strict=True) if not valid
        ],
        "raw_cpm_double_centered_r": double_centered_r(direct, integrated),
        "log1p_double_centered_r": double_centered_r(
            np.log1p(direct), np.log1p(integrated)
        ),
        "raw_mean_within_gene_r": float(np.mean(raw_patterns)),
        "raw_median_within_gene_r": float(np.median(raw_patterns)),
        "log1p_mean_within_gene_r": float(np.mean(log_patterns)),
        "log1p_median_within_gene_r": float(np.median(log_patterns)),
        "raw_cpm_nonzero_fraction": float(np.mean(direct > 0)),
        "integrated_rpkm_nonzero_fraction": float(np.mean(integrated > 0)),
    }


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Zemke 2023 RNA target agreement",
        "",
        "Published reads-per-kilobase-per-million, RPKM, tracks are integrated over sampled union exons and divided by 1,000 to recover counts-per-million scale. Correlation is computed after centering across genes and cell subclasses.",
        "",
        "| Species | Sampled genes | Groups | Raw CPM R | log1p R | Raw within-gene R | log1p within-gene R | Raw nonzero | Integrated nonzero |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["species"]:
        lines.append(
            f"| {row['species']} | {row['sampled_genes']} | {row['groups']} | "
            f"{row['raw_cpm_double_centered_r']:.4f} | {row['log1p_double_centered_r']:.4f} | "
            f"{row['raw_mean_within_gene_r']:.4f} | "
            f"{row['log1p_mean_within_gene_r']:.4f} | "
            f"{row['raw_cpm_nonzero_fraction']:.4f} | "
            f"{row['integrated_rpkm_nonzero_fraction']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--species", nargs="+", default=["human", "macaque", "marmoset", "mouse"])
    parser.add_argument("--sample-genes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--minimum-r", type=float, default=0.7)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    if args.sample_genes < 2:
        parser.error("--sample-genes must be at least two")
    rows = [
        audit_species(
            species,
            args.root / species,
            sample_genes=args.sample_genes,
            seed=args.seed,
        )
        for species in args.species
    ]
    results = {
        "definition": "signed double-centered Pearson correlation over sampled genes and subclasses",
        "minimum_r": args.minimum_r,
        "species": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(results, indent=2) + "\n")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(results))
    failures = [row for row in rows if row["raw_cpm_double_centered_r"] < args.minimum_r]
    if failures:
        summary = ", ".join(
            f"{row['species']}={row['raw_cpm_double_centered_r']:.4f}" for row in failures
        )
        raise SystemExit(f"Raw CPM agreement below {args.minimum_r:.3f}, {summary}")


if __name__ == "__main__":
    main()
