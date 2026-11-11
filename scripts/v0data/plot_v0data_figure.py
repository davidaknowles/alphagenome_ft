#!/usr/bin/env python
"""Panels for the v0data fine-tuning figure: data, example tracks, held-out and eQTL results.

Each panel is written as a vector PDF (and PNG) at its final size; the panels
are assembled with letters by ``docs/figures/v0data_figure.tex``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mizani.breaks import breaks_extended
from plotnine import (
    aes, guide_legend, guides, coord_flip, element_blank, element_line, element_rect, element_text, facet_grid, facet_wrap,
    geom_abline, geom_area, geom_col, geom_hline, geom_line, geom_point, geom_rect,
    geom_segment, geom_text, ggplot, labs, position_dodge, scale_color_manual,
    scale_fill_manual, scale_x_continuous, scale_y_continuous, theme, theme_classic,
)

MODEL_COLORS = {"Head-only": "#2a78d6", "LoRA": "#eb6834", "LoRA+LoCon": "#1baf7a", "Native matched tracks": "#eda100"}
SPECIES_COLORS = {"human": "#e87ba4", "macaque": "#008300", "marmoset": "#6250d6", "mouse": "#e34948"}
OBSERVED = "#9a9a96"
INK = "#2b2b2a"
SOURCE_LABELS = {
    ("hda", "human"): "HDA, first-trimester brain",
    ("liu_hdma", "human"): "HDMA, fetal multi-organ",
    ("johansen2025", "human"): "Johansen, basal ganglia",
    ("johansen2025", "macaque"): "Johansen, basal ganglia",
    ("johansen2025", "marmoset"): "Johansen, basal ganglia",
    ("zemke2023", "human"): "Zemke 2023, motor cortex",
    ("zemke2023", "macaque"): "Zemke 2023, motor cortex",
    ("zemke2023", "marmoset"): "Zemke 2023, motor cortex",
    ("zemke2023", "mouse"): "Zemke 2023, motor cortex",
    ("zemke2024", "human"): "Zemke 2024, hippocampus",
}
GROUPS = {("hda", "human"): (134, 134), ("liu_hdma", "human"): (186, 186), ("zemke2024", "human"): (18, 18)}


def base_theme():
    return theme_classic(base_size=7, base_family="DejaVu Sans") + theme(
        text=element_text(color=INK),
        axis_text=element_text(color=INK, size=6),
        axis_line=element_line(color=INK, size=0.4),
        axis_ticks=element_line(color=INK, size=0.4),
        strip_background=element_blank(),
        strip_text=element_text(size=7, weight="bold"),
        legend_key_size=8,
        legend_text=element_text(size=6),
        legend_title=element_text(size=6.5),
        panel_grid_major_x=element_line(color="#e6e5e0", size=0.3),
        plot_background=element_rect(fill="white", color="white"),
    )


def source_order() -> list[tuple[str, str]]:
    return list(SOURCE_LABELS)


def source_label(dataset: str, source: str) -> str:
    return f"{SOURCE_LABELS[(dataset, source)]} ({source})"


def short_label(dataset: str, source: str) -> str:
    return f"{SOURCE_LABELS[(dataset, source)].split(',')[0]} ({source})"


def panel_data(out: Path) -> None:
    rows = []
    for dataset, source in source_order():
        atac, rna = GROUPS.get((dataset, source), (47, 47) if dataset == "johansen2025" else (20, 19))
        rows += [(source_label(dataset, source), source, "ATAC", atac), (source_label(dataset, source), source, "RNA", rna)]
    data = pd.DataFrame(rows, columns=["label", "species", "assay", "groups"])
    order = [source_label(d, s) for d, s in source_order()][::-1]
    data["label"] = pd.Categorical(data.label, categories=order, ordered=True)
    data["species"] = pd.Categorical(data.species, categories=list(SPECIES_COLORS), ordered=True)
    plot = (
        ggplot(data, aes("label", "groups", fill="species"))
        + geom_col(width=0.7)
        + geom_text(aes(label="groups"), ha="left", nudge_y=5, size=5.5, color=INK)
        + facet_wrap("~assay")
        + scale_fill_manual(values=SPECIES_COLORS, name="Species")
        + scale_y_continuous(limits=(0, 270), breaks=[0, 100, 200], expand=(0, 0))
        + labs(x="", y="Cell groups")
        + base_theme()
        # Row labels name each species, so the fill legend is redundant.
        + theme(axis_text_x=element_text(size=6), axis_text_y=element_text(size=5.8), legend_position="none", panel_spacing_x=0.04)
        + coord_flip()
    )
    save(plot, out / "panel_a", 3.15, 2.6)


def panel_tracks(example: Path, out: Path, groups: list[str], label: str) -> None:
    atac = pd.read_csv(f"{example}_atac.tsv.gz", sep="\t")
    atac = atac[atac.group.isin(groups)].copy()
    atac["mb"] = atac.position / 1e6
    pretty = {group: group.replace("_", "/", 1).replace("_", " ") if group.startswith("L") else group for group in groups}
    atac["track"] = pd.Categorical(atac.group.map(pretty), categories=[pretty[g] for g in groups] + ["Genes"], ordered=True)
    genes = pd.read_csv(f"{example}_genes.tsv.gz", sep="\t")
    keep = ["SLC38A3", "GNAI2", "SEMA3B", "HYAL1", "HYAL2", "TUSC2"]
    genes = genes[genes.gene.isin(keep)]
    bodies = genes.drop_duplicates("gene").copy()
    bodies["y"] = np.where(np.arange(len(bodies)) % 2 == 0, 1.0, 0.0)
    exons = genes.merge(bodies[["gene", "y"]], on="gene")
    track_levels = [pretty[g] for g in groups] + ["Genes"]
    for frame in (bodies, exons):
        frame["track"] = pd.Categorical(["Genes"] * len(frame), categories=track_levels, ordered=True)
    headroom = pd.DataFrame({"mb": [atac.mb.min()], "y": [2.1], "track": pd.Categorical(["Genes"], categories=track_levels, ordered=True)})
    bodies["mid"] = (bodies.start + bodies.end) / 2e6
    long = atac.melt(id_vars=["mb", "track"], value_vars=["observed", "predicted"], var_name="series", value_name="signal")
    long["series"] = long.series.map({"observed": "Observed", "predicted": "Predicted (LoRA)"})

    def y_breaks(limits):
        # The gene track has no meaningful y scale.
        return [] if limits[1] < 2.5 else breaks_extended(n=3)(limits)

    plot = (
        ggplot()
        + geom_area(long[long.series == "Observed"], aes("mb", "signal", fill="series"), alpha=0.85)
        + geom_line(long[long.series != "Observed"], aes("mb", "signal", color="series"), size=0.4)
        + scale_fill_manual(values={"Observed": OBSERVED}, name="")
        + scale_color_manual(values={"Predicted (LoRA)": MODEL_COLORS["LoRA"]}, name="")
        + scale_y_continuous(breaks=y_breaks)
        + geom_segment(bodies, aes(x="start/1e6", xend="end/1e6", y="y", yend="y"), color=INK, size=0.4)
        + geom_rect(exons, aes(xmin="exon_start/1e6", xmax="exon_end/1e6", ymin="y-0.22", ymax="y+0.22"), fill=INK)
        + geom_text(bodies, aes("mid", "y+0.62", label="gene"), size=5.2, fontstyle="italic", color=INK)
        + geom_point(headroom, aes("mb", "y"), alpha=0)
        + facet_grid("track ~ .", scales="free_y")
        + scale_x_continuous(expand=(0, 0))
        + labs(x=f"{label} (Mb)", y="ATAC signal (RPKM)")
        + base_theme()
        + theme(strip_text_y=element_text(angle=0, ha="left", size=6.5, weight="normal"), panel_grid_major_x=element_blank(), axis_text_y=element_text(size=5), legend_position="top", legend_title=element_blank(), legend_box_margin=0)
    )
    save(plot, out / "panel_b", 4.55, 2.9)


def panel_rna(example: Path, out: Path, genes: list[str]) -> None:
    rna = pd.read_csv(f"{example}_rna.tsv.gz", sep="\t")
    rna = rna[rna.gene.isin(genes) & rna.valid].copy()
    rna["obs"] = np.log10(1 + rna.observed_cpm)
    rna["pred"] = np.log10(1 + rna.predicted_cpm)
    rna["gene"] = pd.Categorical(rna.gene, categories=genes, ordered=True)
    stats = rna.groupby("gene", observed=True).apply(lambda d: pd.Series({"r": np.corrcoef(d.obs, d.pred)[0, 1]})).reset_index()
    stats["text"] = stats.r.map(lambda v: f"r = {v:.2f}")
    top = rna.loc[rna.groupby("gene", observed=True).observed_cpm.idxmax()]
    limit = float(max(rna.obs.max(), rna.pred.max())) * 1.08
    plot = (
        ggplot(rna, aes("pred", "obs"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="#b5b4ae", size=0.3)
        + geom_point(color=MODEL_COLORS["LoRA"], size=1.3, stroke=0)
        + geom_text(top, aes(label="group"), ha="right", va="bottom", nudge_x=-0.05, size=5.5, color=INK)
        + geom_text(stats, aes(label="text"), x=0.05, y=limit * 0.93, ha="left", size=5.8, color=INK)
        + facet_wrap("~gene", ncol=1)
        + scale_x_continuous(limits=(0, limit))
        + scale_y_continuous(limits=(0, limit))
        + labs(x="Predicted, log10(1 + CPM)", y="Observed, log10(1 + CPM)")
        + base_theme()
        + theme(strip_text=element_text(fontstyle="italic", weight="bold"))
    )
    save(plot, out / "panel_c", 2.45, 2.9)


def panel_heldout(evaluation: Path, out: Path) -> None:
    rows = json.loads(evaluation.read_text())["rows"]
    names = {"head_only": "Head-only", "lora": "LoRA", "lora+locon": "LoRA+LoCon"}
    data = pd.DataFrame([
        {"label": short_label(r["dataset"], r["source"]), "species": r["source"], "model": names[r["strategy"]],
         "assay": r["modality"].upper(), "test_r": r["test_r"]}
        for r in rows
    ])
    order = [short_label(d, s) for d, s in source_order()][::-1]
    data["label"] = pd.Categorical(data.label, categories=order, ordered=True)
    data["model"] = pd.Categorical(data.model, categories=list(names.values()), ordered=True)
    means = data.groupby(["assay", "model"], observed=True).test_r.mean().reset_index()
    plot = (
        ggplot(data, aes("label", "test_r", color="model"))
        + geom_hline(means, aes(yintercept="test_r", color="model"), linetype="dashed", size=0.35, alpha=0.8)
        + geom_point(position=position_dodge(width=0.6), size=1.4, stroke=0)
        + facet_wrap("~assay")
        + scale_color_manual(values={k: MODEL_COLORS[k] for k in names.values()}, name="Model")
        + scale_y_continuous(limits=(0, 0.85), breaks=[0, 0.2, 0.4, 0.6, 0.8])
        + labs(x="", y="Test R (held-out folds)")
        + base_theme()
        + theme(legend_position="bottom", axis_text_y=element_text(size=5.8), panel_spacing_x=0.12)
        + coord_flip()
    )
    save(plot, out / "panel_d", 3.95, 2.6)


def panel_eqtl(summaries: dict[str, Path], out: Path) -> None:
    order = ["0-1kb", "1-10kb", "10-50kb", "50-100kb", "100-250kb", "250-500kb", ">500kb"]
    names = {"head_only": "Head-only", "lora": "LoRA", "locon": "LoRA+LoCon", "native_matched": "Native matched tracks"}
    frames = [pd.read_csv(path, sep="\t").assign(dataset=name) for name, path in summaries.items()]
    data = pd.concat(frames)
    data = data[(data["mode"] == "gene") & data.distance_bin.isin(order) & data.model.isin(names)].copy()
    data["bin"] = data.distance_bin.map({b: i for i, b in enumerate(order)})
    data["model"] = pd.Categorical(data.model.map(names), categories=list(names.values()), ordered=True)
    data["dataset"] = pd.Categorical(data.dataset, categories=list(summaries), ordered=True)
    plot = (
        ggplot(data, aes("bin", "auroc", color="model"))
        + geom_hline(yintercept=0.5, linetype="dashed", color="#b5b4ae", size=0.3)
        + geom_line(size=0.6)
        + geom_point(size=1.3, stroke=0)
        + facet_wrap("~dataset")
        + scale_color_manual(values=MODEL_COLORS, name="Model")
        + scale_x_continuous(breaks=list(range(len(order))), labels=[b.replace("kb", " kb") for b in order])
        + labs(x="Distance from variant to TSS", y="Gene-span eQTL AUROC")
        + base_theme()
        + theme(axis_text_x=element_text(rotation=30, ha="right"), legend_position="right")
    )
    save(plot, out / "panel_e", 7.1, 2.1)


def save(plot, stem: Path, width: float, height: float) -> None:
    plot.save(stem.with_suffix(".pdf"), width=width, height=height, verbose=False)
    plot.save(stem.with_suffix(".png"), width=width, height=height, dpi=300, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True, help="Stem of predict_example_windows.py outputs")
    parser.add_argument("--example-label", default="Chromosome 3 position")
    parser.add_argument("--evaluation", type=Path, required=True, help="Collated held-out evaluation JSON")
    parser.add_argument("--eqtl-summary", action="append", required=True, help="NAME=summary.tsv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_data(args.output_dir)
    panel_tracks(args.example, args.output_dir, ["Endo", "ODC", "MGC", "ASC", "L2_3_IT"], args.example_label)
    panel_rna(args.example, args.output_dir, ["HYAL2", "SEMA3B"])
    panel_heldout(args.evaluation, args.output_dir)
    panel_eqtl(dict(v.split("=", 1) for v in args.eqtl_summary), args.output_dir)


if __name__ == "__main__":
    main()
