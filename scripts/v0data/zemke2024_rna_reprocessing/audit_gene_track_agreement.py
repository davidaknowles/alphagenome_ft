#!/usr/bin/env python3
"""Compare supported Zemke 2024 raw CPM targets with published RNA tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v0data.zemke2023_rna_reprocessing.audit_gene_track_agreement import (
    _integrate_track,
    double_centered_r,
)


def audit_targets(output_dir: Path, *, sample_genes: int, seed: int) -> dict[str, Any]:
    """Audit raw and published expression over channels with released cell labels."""
    manifest = json.loads((output_dir / "manifest.json").read_text())
    targets = json.loads(Path(manifest["targets"]).read_text())
    rna_heads = [head for head in targets["heads"] if head.get("kind") == "rna_seq"]
    if len(rna_heads) != 1:
        raise ValueError(f"Expected one RNA-seq head, found {len(rna_heads)}.")
    rna_head = rna_heads[0]
    target_groups = tuple(str(target["label"]) for target in rna_head["targets"])
    with np.load(manifest["gene_supervision"]) as supervision:
        groups = tuple(str(value) for value in supervision["groups"])
        group_valid = np.asarray(supervision["group_valid"], dtype=bool)
        cpm = np.asarray(supervision["cpm"], dtype=np.float64)
        chromosomes = supervision["chromosomes"].astype(str)
        exon_offsets = np.asarray(supervision["exon_offsets"], dtype=np.int64)
        exon_starts = np.asarray(supervision["exon_starts"], dtype=np.int64)
        exon_ends = np.asarray(supervision["exon_ends"], dtype=np.int64)
    if groups != target_groups or group_valid.shape != (len(groups),):
        raise ValueError("Gene groups or validity do not match the published RNA target order.")
    eligible = np.flatnonzero((np.diff(exon_offsets) > 0) & np.any(cpm > 0, axis=0))
    if len(eligible) < 2:
        raise ValueError("At least two expressed annotated genes are required.")
    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(eligible, size=min(sample_genes, len(eligible)), replace=False)
    )
    valid_target_indices = np.flatnonzero(group_valid)
    integrated = np.stack(
        [
            _integrate_track(
                Path(rna_head["targets"][group_idx]["path"]),
                chromosomes=chromosomes,
                exon_offsets=exon_offsets,
                exon_starts=exon_starts,
                exon_ends=exon_ends,
                indices=indices,
            )
            for group_idx in valid_target_indices
        ],
        axis=1,
    )
    direct = cpm[group_valid][:, indices].T
    return {
        "sampled_genes": len(indices),
        "published_groups": len(groups),
        "direct_gene_groups": int(group_valid.sum()),
        "masked_gene_groups": [group for group, valid in zip(groups, group_valid) if not valid],
        "raw_cpm_double_centered_r": double_centered_r(direct, integrated),
        "log1p_double_centered_r": double_centered_r(
            np.log1p(direct), np.log1p(integrated)
        ),
        "raw_cpm_nonzero_fraction": float(np.mean(direct > 0)),
        "integrated_rpkm_nonzero_fraction": float(np.mean(integrated > 0)),
    }


def render_markdown(result: dict[str, Any]) -> str:
    return (
        "# Zemke 2024 RNA target agreement\n\n"
        "Published reads-per-kilobase-per-million, RPKM, tracks are integrated over "
        "sampled union exons and divided by 1,000 to recover counts-per-million scale. "
        "Correlation is computed after centering across genes and the 18 broad subclasses "
        "with released cell assignments.\n\n"
        "| Sampled genes | Direct groups | Raw CPM R | log1p R | Raw nonzero | "
        "Integrated nonzero |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"| {result['sampled_genes']} | {result['direct_gene_groups']} | "
        f"{result['raw_cpm_double_centered_r']:.4f} | "
        f"{result['log1p_double_centered_r']:.4f} | "
        f"{result['raw_cpm_nonzero_fraction']:.4f} | "
        f"{result['integrated_rpkm_nonzero_fraction']:.4f} |\n\n"
        "Masked direct-gene channels, " + ", ".join(result["masked_gene_groups"]) + ".\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-genes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--minimum-r", type=float, default=0.7)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_targets(args.output_dir, sample_genes=args.sample_genes, seed=args.seed)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(result))
    if result["raw_cpm_double_centered_r"] < args.minimum_r:
        raise SystemExit(
            f"Raw CPM agreement {result['raw_cpm_double_centered_r']:.4f} is below "
            f"{args.minimum_r:.4f}."
        )


if __name__ == "__main__":
    main()
