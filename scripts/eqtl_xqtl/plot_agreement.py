#!/usr/bin/env python
"""Summarize overlap and PIP agreement between the two fine-mapping sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_abline,
    geom_bin2d,
    geom_col,
    ggplot,
    labs,
    position_dodge,
    scale_fill_gradient,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
    element_text,
)
from scipy.stats import pearsonr, spearmanr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overlap = pd.read_csv(args.input_dir / "finemap_overlap.tsv", sep="\t")
    pairs = pd.read_csv(args.input_dir / "shared_pip_sample.tsv", sep="\t")
    summary = []
    for panel, frame in pairs.groupby("panel", sort=False):
        x = frame.xqtl_pip.to_numpy()
        y = frame.singlebrain_pip.to_numpy()
        summary.append({
            "panel": panel,
            "shared_sampled_gene_variants": len(frame),
            "pearson_r": pearsonr(x, y).statistic if len(frame) > 1 else np.nan,
            "spearman_r": spearmanr(x, y).statistic if len(frame) > 1 else np.nan,
            "xqtl_high_fraction": float((x > 0.75).mean()) if len(frame) else np.nan,
            "singlebrain_high_fraction": float((y > 0.75).mean()) if len(frame) else np.nan,
        })
    agreement = overlap.merge(pd.DataFrame(summary), on="panel", how="left", validate="one_to_one")
    agreement.to_csv(args.output_dir / "finemap_agreement.tsv", sep="\t", index=False)

    label_map = {
        "Ast": "Astrocyte", "Ext": "Excitatory", "IN": "Inhibitory",
        "MG": "Microglia", "OPC": "OPC", "OD": "Oligodendrocyte",
    }
    overlap["cell_type"] = overlap.panel.map(label_map)
    bars = overlap.melt(
        id_vars=["cell_type"],
        value_vars=["xqtl_high_pip", "singlebrain_high_pip", "shared_high_pip"],
        var_name="set_name", value_name="gene_variant_count",
    )
    bars["set_name"] = bars.set_name.map({
        "xqtl_high_pip": "XQTL high-PIP",
        "singlebrain_high_pip": "Singlebrain high-PIP",
        "shared_high_pip": "Shared high-PIP",
    })
    bars["cell_type"] = pd.Categorical(bars.cell_type, categories=list(label_map.values()), ordered=True)
    p_overlap = (
        ggplot(bars, aes("cell_type", "gene_variant_count", fill="set_name"))
        + geom_col(position=position_dodge(width=0.8))
        + labs(x="Cell type", y="High-confidence gene-variant calls", fill="Set")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=30, hjust=1))
    )
    p_overlap.save(args.output_dir / "eqtl_xqtl_finemap_high_pip_overlap.pdf", width=9, height=5, verbose=False)
    p_overlap.save(args.output_dir / "eqtl_xqtl_finemap_high_pip_overlap.png", width=9, height=5, dpi=180, verbose=False)

    pairs["cell_type"] = pairs.panel.map(label_map)
    pairs["cell_type"] = pd.Categorical(pairs.cell_type, categories=list(label_map.values()), ordered=True)
    pairs["xqtl_log10_pip"] = np.log10(np.maximum(pairs.xqtl_pip, 1e-6))
    pairs["singlebrain_log10_pip"] = np.log10(np.maximum(pairs.singlebrain_pip, 1e-6))
    p_pip = (
        ggplot(pairs, aes("xqtl_log10_pip", "singlebrain_log10_pip"))
        + geom_bin2d(bins=45)
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="#555555")
        + scale_x_continuous(limits=(-6, 0), expand=(0, 0))
        + scale_y_continuous(limits=(-6, 0), expand=(0, 0))
        + scale_fill_gradient(low="#f5f5f5", high="#176b87", name="Pairs")
        + labs(x="log10(XQTL PIP; floor 1e-6)", y="log10(singlebrain PIP; floor 1e-6)")
        + theme_bw()
        + theme(figure_size=(10, 7))
    )
    from plotnine import facet_wrap
    p_pip = p_pip + facet_wrap("~cell_type", ncol=3)
    p_pip.save(args.output_dir / "eqtl_xqtl_finemap_pip_agreement.pdf", width=10, height=7, verbose=False)
    p_pip.save(args.output_dir / "eqtl_xqtl_finemap_pip_agreement.png", width=10, height=7, dpi=180, verbose=False)
    print(f"wrote agreement table and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
