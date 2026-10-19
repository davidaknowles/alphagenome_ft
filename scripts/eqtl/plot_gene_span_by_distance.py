#!/usr/bin/env python
"""Plot gene-span eQTL AUROC by TSS distance for several benchmark summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotnine import aes, element_text, facet_wrap, geom_hline, geom_line, geom_point, ggplot, labs, scale_y_continuous, theme, theme_bw

BIN_ORDER = ["0-1kb", "1-10kb", "10-50kb", "50-100kb", "100-250kb", "250-500kb", ">500kb"]
MODEL_LABELS = {
    "head_only": "Head-only",
    "lora": "LoRA",
    "locon": "LoRA+LoCon",
    "raw_locon": "LoRA+LoCon, raw Zemke 2024 ATAC",
    "native_matched": "Native matched tracks",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True, help="NAME=summary.tsv written by plot.py")
    parser.add_argument("--output", type=Path, required=True, help="Output stem; .pdf and .png are written")
    args = parser.parse_args()
    frames = []
    for value in args.summary:
        name, _, path = value.partition("=")
        frames.append(pd.read_csv(path, sep="\t").assign(dataset=name))
    data = pd.concat(frames)
    data = data[(data["mode"] == "gene") & (data.distance_bin != "all")].copy()
    data["distance_bin"] = pd.Categorical(data.distance_bin, categories=BIN_ORDER, ordered=True)
    data["model_label"] = pd.Categorical(data.model.map(MODEL_LABELS), categories=list(MODEL_LABELS.values()), ordered=True)
    data["dataset"] = pd.Categorical(data.dataset, categories=[v.partition("=")[0] for v in args.summary], ordered=True)
    plot = (
        ggplot(data, aes("distance_bin", "auroc", color="model_label", group="model_label"))
        + geom_hline(yintercept=0.5, linetype="dashed", color="#808080")
        + geom_line()
        + geom_point()
        + facet_wrap("~dataset")
        + scale_y_continuous(name="Gene-span AUROC")
        + labs(x="Distance from TSS", color="Model")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot.save(args.output.with_suffix(".pdf"), width=8.5, height=3.8, verbose=False)
    plot.save(args.output.with_suffix(".png"), width=8.5, height=3.8, dpi=180, verbose=False)


if __name__ == "__main__":
    main()
