#!/usr/bin/env python
"""Plot exact high-confidence overlap between the two fine-mapping sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    geom_col,
    ggplot,
    labs,
    position_dodge,
    theme,
    theme_bw,
    element_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overlap = pd.read_csv(args.input_dir / "finemap_overlap.tsv", sep="\t")
    overlap = overlap.drop(columns=["sampled_shared_pips", "xqtl_sampled_rows"], errors="ignore")
    overlap.to_csv(args.output_dir / "finemap_agreement.tsv", sep="\t", index=False)

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

    print(f"wrote exact-overlap table and plot to {args.output_dir}")


if __name__ == "__main__":
    main()
