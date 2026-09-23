#!/usr/bin/env python
"""Summarize and plot eQTL benchmark scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotnine import aes, facet_wrap, geom_hline, geom_line, geom_point, ggplot, labs, theme_bw, theme, element_text, scale_y_continuous
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores, sep="\t")
    rows = []
    for (model, mode, distance_bin), frame in scores.assign(mode="tss").groupby(["model", "mode", "distance_bin"], sort=False):
        if frame.label.nunique() < 2:
            continue
        values = frame.tss_effect
        rows.append({"model": model, "mode": mode, "distance_bin": distance_bin, "auroc": roc_auc_score(frame.label, values), "auprc": average_precision_score(frame.label, values), "n_positive": int(frame.label.sum()), "n_negative": int((1 - frame.label).sum())})
    for (model, mode, distance_bin), frame in scores.assign(mode="gene").groupby(["model", "mode", "distance_bin"], sort=False):
        if frame.label.nunique() < 2:
            continue
        values = frame.gene_effect
        rows.append({"model": model, "mode": mode, "distance_bin": distance_bin, "auroc": roc_auc_score(frame.label, values), "auprc": average_precision_score(frame.label, values), "n_positive": int(frame.label.sum()), "n_negative": int((1 - frame.label).sum())})
    for (model, mode), frame in scores.assign(mode="tss").groupby(["model", "mode"], sort=False):
        rows.append({"model": model, "mode": mode, "distance_bin": "all", "auroc": roc_auc_score(frame.label, frame.tss_effect), "auprc": average_precision_score(frame.label, frame.tss_effect), "n_positive": int(frame.label.sum()), "n_negative": int((1 - frame.label).sum())})
    for (model, mode), frame in scores.assign(mode="gene").groupby(["model", "mode"], sort=False):
        rows.append({"model": model, "mode": mode, "distance_bin": "all", "auroc": roc_auc_score(frame.label, frame.gene_effect), "auprc": average_precision_score(frame.label, frame.gene_effect), "n_positive": int(frame.label.sum()), "n_negative": int((1 - frame.label).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)
    bin_order = ["0-1kb", "1-10kb", "10-50kb", "50-100kb", "100-250kb", "250-500kb", ">500kb"]
    plot_summary = summary[summary["distance_bin"] != "all"].copy()
    plot_summary["distance_bin"] = pd.Categorical(plot_summary["distance_bin"], categories=bin_order, ordered=True)
    plot_summary["model_label"] = plot_summary.model.map({"head_only": "Head-only", "lora": "LoRA", "locon": "LoRA+LoCon"})
    plot_summary["mode_label"] = plot_summary["mode"].map({"tss": "TSS bin", "gene": "Gene span"})
    random_summary = (plot_summary[plot_summary["mode"] == "tss"]
                      [["distance_bin", "n_positive", "n_negative"]]
                      .drop_duplicates("distance_bin").copy())
    random_summary["random_ap"] = random_summary.n_positive / (random_summary.n_positive + random_summary.n_negative)
    def distance_plot(metric: str):
        return (ggplot(plot_summary, aes("distance_bin", metric, color="model_label", group="model_label")) + geom_line() + geom_point() + facet_wrap("~mode_label") + labs(x="Distance from TSS", color="Model") + theme_bw() + theme(axis_text_x=element_text(rotation=45, hjust=1)))

    auroc = distance_plot("auroc") + geom_hline(yintercept=0.5, linetype="dashed", color="#808080") + scale_y_continuous(limits=(0.4, 1.0), name="AUROC")
    aupr = (distance_plot("auprc")
            + geom_point(data=random_summary, mapping=aes("distance_bin", "random_ap"), inherit_aes=False, color="#222222", shape="D", size=2.2)
            + scale_y_continuous(limits=(0.0, 1.0), name="AUPR"))
    auroc.save(args.output_dir / "eqtl_auroc_by_distance.pdf", width=8, height=4.5, verbose=False)
    auroc.save(args.output_dir / "eqtl_auroc_by_distance.png", width=8, height=4.5, dpi=180, verbose=False)
    aupr.save(args.output_dir / "eqtl_aupr_by_distance.pdf", width=8, height=4.5, verbose=False)
    aupr.save(args.output_dir / "eqtl_aupr_by_distance.png", width=8, height=4.5, dpi=180, verbose=False)


if __name__ == "__main__":
    main()
