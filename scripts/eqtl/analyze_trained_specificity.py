#!/usr/bin/env python
"""Test whether trained Zemke 2023 RNA tracks are cell-type specific for eQTLs.

Each eQTL panel is scored with every RNA track and with every cell-group
composite (the mean effect over a group's tracks, as in the benchmark). If the
tracks carry cell-type-specific signal, a panel should be best discriminated by
its own group's composite. The diagonal statistic is the mean matched AUROC
minus the mean unmatched AUROC across panels, and its p-value comes from all
permutations of the composite-to-panel assignment.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import aes, element_text, facet_grid, geom_text, geom_tile, ggplot, labs, scale_color_manual, scale_fill_gradient2, theme, theme_bw
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_native_specificity import _metric  # noqa: E402
from benchmark import GROUP_TRACKS  # noqa: E402

SUMMARIES = {"tss": "TSS bin", "gene": "Gene span"}
MODEL_LABELS = {"head_only": "Head-only", "lora": "LoRA", "locon": "LoRA+LoCon", "raw_locon": "LoRA+LoCon, raw Zemke 2024 ATAC"}
GROUP_ORDER = list(GROUP_TRACKS)
UNTRAINED_TRACKS = ("L5_IT",)


def load_model_effects(path: Path) -> tuple[str, pd.DataFrame, dict[str, np.ndarray], list[str]]:
    """Read one model's per-track effects and the score rows they align to."""
    model = path.stem.removeprefix("track_effects_")
    scores = pd.read_csv(path.parent / f"scores_{model}.tsv", sep="\t")
    effects = np.load(path)
    if len(scores) != len(effects["tss"]) or not (
        np.array_equal(scores.panel.to_numpy(), effects["panel"]) and np.array_equal(scores.pos.to_numpy(), effects["pos"])
    ):
        raise ValueError(f"{path} is not aligned with scores_{model}.tsv")
    return model, scores, {summary: effects[summary] for summary in SUMMARIES}, [str(t) for t in effects["tracks"]]


def composite_effects(values: np.ndarray, tracks: list[str]) -> dict[str, np.ndarray]:
    position = {track: index for index, track in enumerate(tracks)}
    return {group: values[:, [position[t] for t in members]].mean(axis=1) for group, members in GROUP_TRACKS.items()}


def diagonal_test(matrix: pd.DataFrame) -> tuple[float, float]:
    """Matched-minus-unmatched mean AUROC and its exact permutation p-value."""
    values = matrix.to_numpy()
    k = values.shape[0]

    def statistic(order: tuple[int, ...]) -> float:
        matched = values[np.arange(k), order]
        return float(matched.mean() - (values.sum() - matched.sum()) / (k * k - k))

    observed = statistic(tuple(range(k)))
    null = np.asarray([statistic(order) for order in itertools.permutations(range(k))])
    return observed, float(np.mean(null >= observed - 1e-12))


def analyze(datasets: dict[str, Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    track_rows, composite_rows, test_rows, correlation_rows = [], [], [], []
    for dataset, root in datasets.items():
        for path in sorted(root.rglob("track_effects_*.npz")):
            model, scores, effects, tracks = load_model_effects(path)
            trained = [t for t in tracks if t not in UNTRAINED_TRACKS]
            labels = scores.label.to_numpy()
            for summary, values in effects.items():
                trained_values = values[:, [tracks.index(t) for t in trained]]
                rho = spearmanr(trained_values).statistic
                correlation_rows.append({
                    "dataset": dataset, "model": model, "summary": summary,
                    "median_track_spearman": float(np.median(rho[np.triu_indices(len(trained), 1)])),
                })
                composites = composite_effects(values, tracks)
                panel_groups = [g for g in GROUP_ORDER if g in set(scores.cell_group)]
                for panel_group in panel_groups:
                    mask = (scores.cell_group == panel_group).to_numpy()
                    base = {"dataset": dataset, "model": model, "summary": summary, "panel_group": panel_group,
                            "n_positive": int(labels[mask].sum()), "n_negative": int((1 - labels[mask]).sum())}
                    for track in trained:
                        auroc, aupr = _metric(labels[mask], values[mask, tracks.index(track)])
                        track_rows.append({**base, "track": track, "matched": track in GROUP_TRACKS[panel_group], "auroc": auroc, "aupr": aupr})
                    for track_group, composite in composites.items():
                        auroc, aupr = _metric(labels[mask], composite[mask])
                        composite_rows.append({**base, "track_group": track_group, "matched": track_group == panel_group, "auroc": auroc, "aupr": aupr})
                matrix = (pd.DataFrame(composite_rows)
                          .query("dataset == @dataset and model == @model and summary == @summary and track_group in @panel_groups")
                          .pivot(index="panel_group", columns="track_group", values="auroc")
                          .loc[panel_groups, panel_groups])
                observed, p_value = diagonal_test(matrix)
                test_rows.append({"dataset": dataset, "model": model, "summary": summary, "n_groups": len(panel_groups),
                                  "matched_minus_unmatched_auroc": observed, "permutation_p": p_value})
    tracks_df, composites_df = pd.DataFrame(track_rows), pd.DataFrame(composite_rows)
    tests_df = pd.DataFrame(test_rows).merge(pd.DataFrame(correlation_rows), on=["dataset", "model", "summary"])
    tracks_df.to_csv(output_dir / "trained_track_metrics.tsv", sep="\t", index=False)
    composites_df.to_csv(output_dir / "trained_group_metrics.tsv", sep="\t", index=False)
    tests_df.to_csv(output_dir / "trained_specificity_tests.tsv", sep="\t", index=False)
    _write_ranks(tracks_df, composites_df, output_dir)
    for dataset in datasets:
        _plot_matrix(composites_df[composites_df.dataset == dataset], output_dir / f"trained_specificity_{dataset}")
    print(tests_df.to_string(index=False))


def _write_ranks(tracks: pd.DataFrame, composites: pd.DataFrame, output_dir: Path) -> None:
    """Rank of the matched composite among composites, and of the best matched track among tracks."""
    keys = ["dataset", "model", "summary", "panel_group"]
    composites = composites[composites.track_group.isin(composites.panel_group.unique())].copy()
    composites["rank"] = composites.groupby(keys).auroc.rank(ascending=False)
    tracks = tracks.copy()
    tracks["rank"] = tracks.groupby(keys).auroc.rank(ascending=False)
    best = tracks.loc[tracks.groupby(keys).auroc.idxmax(), keys + ["track", "auroc"]].rename(columns={"track": "best_track", "auroc": "best_track_auroc"})
    matched_track = tracks[tracks.matched].groupby(keys).agg(best_matched_track_rank=("rank", "min")).reset_index()
    ranks = (composites[composites.matched][keys + ["auroc", "rank"]]
             .rename(columns={"auroc": "matched_group_auroc", "rank": "matched_group_rank"})
             .merge(composites[~composites.matched].groupby(keys).auroc.max().rename("best_unmatched_group_auroc").reset_index(), on=keys)
             .merge(matched_track, on=keys).merge(best, on=keys))
    ranks["n_tracks"] = tracks.groupby(keys).size().iloc[0]
    ranks.to_csv(output_dir / "trained_specificity_ranks.tsv", sep="\t", index=False)


def _plot_matrix(composites: pd.DataFrame, stem: Path) -> None:
    groups = [g for g in GROUP_ORDER if g in set(composites.panel_group)]
    data = composites[composites.track_group.isin(groups)].copy()
    data["delta"] = data.auroc - data.groupby(["model", "summary", "panel_group"]).auroc.transform("mean")
    data["panel_group"] = pd.Categorical(data.panel_group, categories=groups[::-1], ordered=True)
    data["track_group"] = pd.Categorical(data.track_group, categories=groups, ordered=True)
    models = [m for m in MODEL_LABELS if m in set(data.model)] + sorted(set(data.model) - set(MODEL_LABELS))
    data["model_label"] = pd.Categorical(data.model.map(lambda m: MODEL_LABELS.get(m, m)), categories=[MODEL_LABELS.get(m, m) for m in models], ordered=True)
    data["summary_label"] = pd.Categorical(data.summary.map(SUMMARIES), categories=list(SUMMARIES.values()), ordered=True)
    data["text"] = data.auroc.map("{:.2f}".format)
    limit = float(np.nanmax(np.abs(data.delta)))
    plot = (
        ggplot(data, aes("track_group", "panel_group", fill="delta"))
        + geom_tile(aes(color="matched"), size=0.6)
        + geom_text(aes(label="text"), size=6)
        + facet_grid("summary_label ~ model_label")
        + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac", limits=(-limit, limit), name="AUROC minus\npanel mean")
        + labs(x="Composite of model tracks", y="eQTL panel", color="Matched")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )
    plot = plot + scale_color_manual(values={True: "#000000", False: "#ffffff"}, guide=None)
    width = 3 + 2.6 * data.model.nunique()
    plot.save(stem.with_suffix(".pdf"), width=width, height=6.5, verbose=False)
    plot.save(stem.with_suffix(".png"), width=width, height=6.5, dpi=180, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="NAME=DIR containing track_effects_*.npz, searched recursively")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    datasets = {}
    for value in args.dataset:
        name, _, root = value.partition("=")
        datasets[name] = Path(root)
    analyze(datasets, args.output_dir)


if __name__ == "__main__":
    main()
