#!/usr/bin/env python
"""Rank native AlphaGenome RNA-seq and CAGE tracks by eQTL-panel predictivity."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    facet_grid,
    geom_point,
    ggplot,
    labs,
    position_jitter,
    scale_y_reverse,
    theme_bw,
)
from sklearn.metrics import average_precision_score, roc_auc_score


PANEL_PREFIXES = ("Ast", "End", "Ext", "IN", "MG", "OD", "OPC")
PANEL_LABELS = {
    "Ast": "astrocyte",
    "End": "endothelial",
    "Ext": "excitatory",
    "IN": "inhibitory",
    "MG": "microglia",
    "OD": "oligodendrocyte",
    "OPC": "oligodendrocyte precursor",
}


def _proxy_tier(group: str, output_type: str, track_text: str) -> str:
    if group == "Ast" and "astrocyte" in track_text:
        return "direct_cell_label"
    if group == "End" and "endothelial" in track_text:
        return "direct_cell_label"
    if group == "Ext" and "glutamatergic neuron" in track_text:
        return "related_neuronal_cell"
    if group == "IN" and re.search(r"\bneuron\b|neuronal stem", track_text):
        return "generic_neuronal_proxy"
    if group == "MG" and "monocyte" in track_text:
        return "microglia_proxy_cd14_monocyte"
    if group in {"OD", "OPC"}:
        if output_type == "cage" and "oligodendrocyte precursor cell" in track_text:
            return "direct_opc" if group == "OPC" else "immature_oligo_proxy"
        if re.search(r"\bbrain\b|cortex|cerebell", track_text):
            return "bulk_brain_proxy"
    return "unmatched"


def _metric(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(score)
    y = y_true[valid]
    s = score[valid]
    if y.size == 0 or np.unique(y).size != 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def analyze(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = pd.read_csv(input_dir / "variants.tsv", sep="\t")
    tracks = pd.read_csv(input_dir / "tracks.tsv", sep="\t").fillna("")
    completion = np.load(input_dir / "completion.npy", mmap_mode="r")
    if not bool(np.all(completion)):
        raise RuntimeError(f"Native VEP matrix is incomplete: {int(completion.sum())}/{completion.size} records")
    effects = np.load(input_dir / "native_vep_delta.npy", mmap_mode="r")
    if effects.shape != (len(variants), 2, len(tracks)):
        raise ValueError(f"VEP matrix shape {effects.shape} does not match variants/tracks")

    variants["panel_group"] = variants["panel"].str.extract(r"^(Ast|End|Ext|IN|MG|OD|OPC)", expand=False)
    track_index = {track_id: i for i, track_id in enumerate(tracks["track_id"])}
    grouped_tracks = tracks.groupby(
        ["output_type", "data_source", "biosample_name", "biosample_type", "Assay title"],
        dropna=False,
        sort=False,
    )
    sample_tracks = []
    for sample_key, sample in grouped_tracks:
        output_type = str(sample_key[0])
        channel_indices = np.asarray([track_index[t] for t in sample.track_id], dtype=np.int32)
        sample_tracks.append((sample, output_type, channel_indices))

    rows: list[dict[str, object]] = []
    for panel in PANEL_PREFIXES:
        panel_mask = variants.panel_group.to_numpy() == panel
        panel_variants = variants.loc[panel_mask]
        labels = panel_variants.label.to_numpy(dtype=np.int8)
        if np.unique(labels).size != 2:
            continue
        for sample, output_type, channel_indices in sample_tracks:
            if output_type not in {"rna_seq", "cage"}:
                continue
            metadata = sample.iloc[0]
            text = " ".join(str(metadata.get(key, "") or "").lower() for key in (
                "biosample_name", "biosample_type", "gtex_tissue", "data_source", "Assay title", "ontology_curie"
            ))
            proxy_tier = _proxy_tier(panel, output_type, text)
            for summary_index, summary_name in enumerate(("tss_bin", "gene_span")):
                values = np.abs(effects[panel_mask, summary_index, :][:, channel_indices]).mean(axis=1)
                auroc, aupr = _metric(labels, values)
                rows.append({
                    "panel_group": panel,
                    "panel_label": PANEL_LABELS[panel],
                    "output_type": output_type,
                    "summary": summary_name,
                    "data_source": metadata.get("data_source", ""),
                    "biosample_name": metadata.get("biosample_name", ""),
                    "biosample_type": metadata.get("biosample_type", ""),
                    "assay": metadata.get("Assay title", ""),
                    "channel_count": len(channel_indices),
                    "proxy_tier": proxy_tier,
                    "auroc": auroc,
                    "aupr": aupr,
                    "n_positive": int(labels.sum()),
                    "n_negative": int((1 - labels).sum()),
                })

    metrics = pd.DataFrame(rows)
    metrics["auroc_rank"] = metrics.groupby(["panel_group", "output_type", "summary"]).auroc.rank(ascending=False, method="min")
    metrics["aupr_rank"] = metrics.groupby(["panel_group", "output_type", "summary"]).aupr.rank(ascending=False, method="min")
    metrics.to_csv(output_dir / "native_track_specificity.tsv", sep="\t", index=False)

    summary = metrics.groupby(["panel_group", "panel_label", "output_type", "summary", "proxy_tier"], as_index=False).agg(
        tracks=("biosample_name", "size"),
        median_auroc=("auroc", "median"),
        median_aupr=("aupr", "median"),
        best_auroc=("auroc", "max"),
        best_aupr=("aupr", "max"),
        median_auroc_rank=("auroc_rank", "median"),
        median_aupr_rank=("aupr_rank", "median"),
    )
    summary.to_csv(output_dir / "native_proxy_summary.tsv", sep="\t", index=False)
    candidates = metrics.loc[metrics.proxy_tier != "unmatched"]
    if not candidates.empty:
        rank_plot = (
            ggplot(candidates, aes("panel_group", "aupr_rank", color="proxy_tier"))
            + geom_point(position=position_jitter(width=0.12, height=0), alpha=0.75)
            + facet_grid("summary ~ output_type", scales="free_y")
            + scale_y_reverse()
            + labs(
                x="Fine-mapping panel",
                y="Average-precision rank among native tracks (1 is best)",
                color="Track relationship",
            )
            + theme_bw()
        )
        rank_plot.save(output_dir / "native_proxy_track_ranks.pdf", width=11, height=6, verbose=False)
        rank_plot.save(output_dir / "native_proxy_track_ranks.png", width=11, height=6, dpi=180, verbose=False)
    print(f"wrote {len(metrics)} track-panel metrics to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
