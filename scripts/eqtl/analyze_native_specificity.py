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
    geom_tile,
    ggplot,
    labs,
    position_jitter,
    scale_fill_gradient,
    scale_y_reverse,
    theme_bw,
    theme,
    element_text,
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


def _proxy_tier(group: str, output_type: str, metadata: pd.Series, track_text: str) -> str:
    if group == "Ast" and "astrocyte" in track_text:
        return "direct_cell_label"
    if group == "End" and "endothelial" in track_text:
        return "direct_cell_label"
    if group == "Ext" and "glutamatergic neuron" in track_text:
        return "related_neuronal_cell"
    if group == "IN" and re.search(r"\bneuron\b", track_text) and not re.search(
        r"glutamatergic|motor neuron|neuronal stem", track_text
    ):
        return "generic_neuronal_proxy"
    if group == "MG" and "spleen" in track_text:
        return "myeloid_rich_tissue_proxy"
    if group == "MG" and "monocyte" in track_text:
        return "microglia_proxy_cd14_monocyte"
    if group in {"OD", "OPC"}:
        if output_type == "cage" and "oligodendrocyte precursor cell" in track_text:
            return "direct_opc" if group == "OPC" else "immature_oligo_proxy"
        tissue_name = " ".join(str(metadata.get(key, "") or "").lower() for key in ("biosample_name", "gtex_tissue"))
        is_tissue = str(metadata.get("biosample_type", "")).lower() == "tissue"
        is_brain_region = re.search(
            r"\bbrain\b|cerebell|cerebral|prefrontal|frontal cortex|cingulate|parietal|temporal|"
            r"occipital|spinal cord|basal ganglia|hippocamp|amygdala|hypothalam|substantia nigra|"
            r"putamen|caudate|thalam|globus pallidus|corpus callosum|\bpons\b|midbrain|medulla|olfactory",
            tissue_name,
        )
        is_kidney = re.search(r"kidney|renal", tissue_name)
        if is_tissue and is_brain_region and not is_kidney:
            return "bulk_brain_proxy"
    return "unmatched"


def _metric(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(score)
    y = y_true[valid]
    s = score[valid]
    if y.size == 0 or np.unique(y).size != 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _refresh_proxy_annotations(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    inhibitory = metrics.panel_group == "IN"
    inhibitory_name = metrics.biosample_name.fillna("").str.lower()
    generic_neuron = inhibitory_name.str.contains(r"\bneuron\b") & ~inhibitory_name.str.contains(
        r"glutamatergic|motor neuron|neuronal stem"
    )
    metrics.loc[inhibitory, "proxy_tier"] = "unmatched"
    metrics.loc[inhibitory & generic_neuron, "proxy_tier"] = "generic_neuronal_proxy"
    spleen = (metrics.panel_group == "MG") & metrics.biosample_name.fillna("").str.lower().str.contains("spleen")
    metrics.loc[spleen, "proxy_tier"] = "myeloid_rich_tissue_proxy"
    return metrics


def _write_proxy_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
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


def _plot_specificity(metrics: pd.DataFrame, output_dir: Path) -> None:
    candidates = metrics.loc[metrics.proxy_tier != "unmatched"].copy()
    if candidates.empty:
        return
    candidates["output_label"] = candidates.output_type.map({"rna_seq": "RNA-seq", "cage": "CAGE"})
    candidates["summary_label"] = candidates.summary.map({"tss_bin": "TSS bin", "gene_span": "Gene span"})
    candidates["proxy_label"] = candidates.proxy_tier.map({
        "direct_cell_label": "Direct cell label",
        "related_neuronal_cell": "Related neuronal cell",
        "generic_neuronal_proxy": "Generic neuronal proxy",
        "microglia_proxy_cd14_monocyte": "CD14+ monocyte proxy",
        "myeloid_rich_tissue_proxy": "Myeloid-rich tissue proxy",
        "bulk_brain_proxy": "Bulk brain tissue",
        "direct_opc": "Direct OPC",
        "immature_oligo_proxy": "Immature oligo proxy",
    })
    rank_plot = (
        ggplot(candidates, aes("panel_group", "aupr_rank", color="proxy_label"))
        + geom_point(position=position_jitter(width=0.12, height=0), alpha=0.75)
        + facet_grid("summary_label ~ output_label", scales="free_y")
        + scale_y_reverse()
        + labs(
            title="Ranks of native cell-type candidates across eQTL panels",
            x="Fine-mapping panel",
            y="Average-precision rank among native tracks (1 is best)",
            color="Candidate track",
        )
        + theme_bw()
    )
    rank_plot.save(output_dir / "native_proxy_track_ranks.pdf", width=11, height=6, verbose=False)
    rank_plot.save(output_dir / "native_proxy_track_ranks.png", width=11, height=6, dpi=180, verbose=False)

    track_keys = ["output_type", "data_source", "biosample_name", "biosample_type", "assay"]
    candidates["track_id"] = candidates[track_keys].astype(str).agg(" | ".join, axis=1)
    top_ids = set().union(*(
        set(group.nlargest(5, "aupr").track_id)
        for _, group in candidates.groupby(["panel_group", "summary"])
    ))
    heatmap_data = metrics.copy()
    heatmap_data["track_id"] = heatmap_data[track_keys].astype(str).agg(" | ".join, axis=1)
    heatmap_data = heatmap_data[heatmap_data.track_id.isin(top_ids)].copy()
    track_rows = heatmap_data[track_keys + ["track_id"]].drop_duplicates().copy()
    assay_labels = {"polyA plus RNA-seq": "polyA", "total RNA-seq": "total", "hCAGE": "hCAGE", "LQhCAGE": "LQhCAGE"}
    track_rows["track_label"] = track_rows.apply(
        lambda row: f"{'RNA' if row.output_type == 'rna_seq' else 'CAGE'}: {row.biosample_name} ({row.data_source}, {assay_labels.get(row.assay, row.assay)})",
        axis=1,
    )
    heatmap_data = heatmap_data.merge(track_rows[track_keys + ["track_id", "track_label"]], on=track_keys + ["track_id"], how="left")
    label_order = track_rows.sort_values("track_label").track_label.tolist()
    panel_order = ["Ast", "End", "Ext", "IN", "MG", "OD", "OPC"]
    heatmap_data["track_label"] = pd.Categorical(heatmap_data.track_label, categories=label_order, ordered=True)
    heatmap_data["panel_label"] = pd.Categorical(heatmap_data.panel_label, categories=[PANEL_LABELS[p] for p in panel_order], ordered=True)
    heatmap_data["summary_label"] = pd.Categorical(heatmap_data.summary.map({"tss_bin": "TSS bin", "gene_span": "Gene span"}), categories=["TSS bin", "Gene span"], ordered=True)
    heatmap = (
        ggplot(heatmap_data, aes("panel_label", "track_label", fill="aupr"))
        + geom_tile(color="white", size=0.15)
        + facet_grid(". ~ summary_label")
        + scale_fill_gradient(low="#f2f2f2", high="#176b87", limits=(0, 1), name="Average\nprecision")
        + labs(x="eQTL cell type", y="Candidate native track", title="Native AlphaGenome track average precision")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=35, hjust=1, size=9), axis_text_y=element_text(size=8), figure_size=(13, 9))
    )
    heatmap.save(output_dir / "native_candidate_track_ap_heatmap.pdf", width=13, height=9, verbose=False)
    heatmap.save(output_dir / "native_candidate_track_ap_heatmap.png", width=13, height=9, dpi=180, verbose=False)


def _write_biological_matches(input_dir: Path, output_dir: Path, metrics: pd.DataFrame) -> None:
    track_keys = ["output_type", "data_source", "biosample_name", "biosample_type", "assay"]
    candidates = metrics.loc[metrics.proxy_tier != "unmatched"].copy()
    candidates = candidates.pivot_table(
        index=["panel_group", "panel_label", *track_keys, "proxy_tier"],
        columns="summary",
        values="aupr",
    ).reset_index()
    candidates["mean_ap"] = candidates[["tss_bin", "gene_span"]].mean(axis=1)
    matches = (candidates.sort_values(["panel_group", "mean_ap", "biosample_name"], ascending=[True, False, True])
               .drop_duplicates("panel_group", keep="first").copy())
    matches = matches.rename(columns={"tss_bin": "tss_ap", "gene_span": "gene_ap"})
    matches.to_csv(output_dir / "native_biological_track_matches.tsv", sep="\t", index=False)

    variants = pd.read_csv(input_dir / "variants.tsv", sep="\t")
    tracks = pd.read_csv(input_dir / "tracks.tsv", sep="\t").fillna("")
    tracks = tracks.rename(columns={"Assay title": "assay"})
    effects = np.load(input_dir / "native_vep_delta.npy", mmap_mode="r")
    track_index = {track_id: i for i, track_id in enumerate(tracks.track_id)}
    variant_groups = variants.panel.str.extract(r"^(Ast|End|Ext|IN|MG|OD|OPC)", expand=False)
    selected_effects = np.full((len(variants), 2), np.nan, dtype=np.float32)
    track_groups = tracks.groupby(track_keys, dropna=False, sort=False)
    for match in matches.to_dict("records"):
        group = match["panel_group"]
        key = tuple(match[column] for column in track_keys)
        sample = track_groups.get_group(key)
        channel_indices = np.asarray([track_index[track_id] for track_id in sample.track_id], dtype=np.int32)
        variant_indices = np.flatnonzero(variant_groups.to_numpy() == group)
        for summary_index in range(2):
            selected_effects[variant_indices, summary_index] = np.abs(
                effects[variant_indices, summary_index, :][:, channel_indices]
            ).mean(axis=1)

    score_table = variants[["panel", "feature", "chrom", "pos", "label", "distance_bin"]].copy()
    score_table["native_tss_effect"] = selected_effects[:, 0]
    score_table["native_gene_effect"] = selected_effects[:, 1]
    score_table.to_csv(output_dir / "native_biological_track_scores.tsv", sep="\t", index=False)


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
            proxy_tier = _proxy_tier(panel, output_type, metadata, text)
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

    _write_proxy_summary(metrics, output_dir)
    _plot_specificity(metrics, output_dir)
    _write_biological_matches(input_dir, output_dir, metrics)
    print(f"wrote {len(metrics)} track-panel metrics to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true", help="Regenerate the plot from existing track metrics")
    args = parser.parse_args()
    if args.plot_only:
        metrics = pd.read_csv(args.output_dir / "native_track_specificity.tsv", sep="\t")
        metrics = _refresh_proxy_annotations(metrics)
        metrics.to_csv(args.output_dir / "native_track_specificity.tsv", sep="\t", index=False)
        _write_proxy_summary(metrics, args.output_dir)
        _plot_specificity(metrics, args.output_dir)
        _write_biological_matches(args.input_dir, args.output_dir, metrics)
    else:
        analyze(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
