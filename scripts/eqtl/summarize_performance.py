#!/usr/bin/env python
"""Pooled and within-cell-group eQTL discrimination for each model.

Pooled AUROC and AP rank variants from all panels together; the within-group
AUROC is the mean of AUROC computed separately in each panel cell group, which
does not depend on how effect scales differ between track groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

KEY = ["panel", "feature", "chrom", "pos"]


def with_native(scores: pd.DataFrame, native_path: Path | None) -> pd.DataFrame:
    """Append the selected native AlphaGenome tracks as model ``native_matched``."""
    if native_path is None:
        return scores
    base = scores[scores.model == scores.model.iloc[0]].drop_duplicates(KEY)
    native = pd.read_csv(native_path, sep="\t")
    matched = base.merge(native[KEY + ["native_tss_effect", "native_gene_effect"]], on=KEY, how="inner", validate="one_to_one")
    matched = matched.dropna(subset=["native_tss_effect", "native_gene_effect"])
    matched = matched.assign(tss_effect=matched.native_tss_effect, gene_effect=matched.native_gene_effect, model="native_matched")
    return pd.concat([scores, matched[scores.columns]], ignore_index=True)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in scores.groupby("model"):
        row = {"model": model, "n": len(frame)}
        for summary in ("tss", "gene"):
            effect = frame[f"{summary}_effect"]
            row[f"{summary}_auroc"] = roc_auc_score(frame.label, effect)
            row[f"{summary}_ap"] = average_precision_score(frame.label, effect)
            row[f"{summary}_within_auroc"] = float(np.mean([
                roc_auc_score(group.label, group[f"{summary}_effect"]) for _, group in frame.groupby("cell_group")
            ]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="NAME=scores.tsv[=native_scores.tsv]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tables = []
    for value in args.dataset:
        name, scores_path, *native = value.split("=")
        scores = with_native(pd.read_csv(scores_path, sep="\t"), Path(native[0]) if native else None)
        tables.append(summarize(scores).assign(dataset=name))
    table = pd.concat(tables)[["dataset", "model", "n", "tss_auroc", "tss_ap", "tss_within_auroc", "gene_auroc", "gene_ap", "gene_within_auroc"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
