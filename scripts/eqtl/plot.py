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
    parser.add_argument("--tex-table", type=Path, default=None)
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
    if args.tex_table is not None:
        args.tex_table.parent.mkdir(parents=True, exist_ok=True)
        with args.tex_table.open("w") as handle:
            for row in summary.itertuples(index=False):
                model = str(row.model).replace("_", "\\_")
                aggregation = "TSS bin" if row.mode == "tss" else "Gene span"
                handle.write(f"{model} & {aggregation} & {row.distance_bin} & {row.auroc:.3f} & {row.auprc:.3f} & {row.n_positive} & {row.n_negative} \\\\\n")
    bin_order = ["0-1kb", "1-10kb", "10-50kb", "50-100kb", "100-250kb", "250-500kb", ">500kb"]
    plot_summary = summary[summary["distance_bin"] != "all"].copy()
    plot_summary["distance_bin"] = pd.Categorical(plot_summary["distance_bin"], categories=bin_order, ordered=True)
    plot = (ggplot(plot_summary, aes("distance_bin", "auroc", color="model", group="model")) + geom_hline(yintercept=0.5, linetype="dashed", color="#808080") + geom_line() + geom_point() + facet_wrap("~mode") + scale_y_continuous(limits=(0.4, 1.0)) + labs(x="Distance from TSS", y="AUROC", color="Model") + theme_bw() + theme(axis_text_x=element_text(rotation=45, hjust=1)))
    plot.save(args.output_dir / "auroc_by_distance.pdf", width=8, height=4.5, verbose=False)
    plot.save(args.output_dir / "auroc_by_distance.png", width=8, height=4.5, dpi=180, verbose=False)


if __name__ == "__main__":
    main()
