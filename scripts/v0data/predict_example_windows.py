#!/usr/bin/env python
"""Predict held-out windows for one source and save observed and predicted signal.

With ``--window``, writes 128 bp observed and predicted coverage for the source's
ATAC head and gene-level observed and predicted CPM for its RNA head (exon-weighted
readout, as in training) for each window, for figure panels.

With ``--score-split``, predicts every window of that split in the source's
``split_intervals_bed`` (skipping windows that overlap ``exclude_intervals_bed``)
and writes per-window double-centered ATAC R and per-gene RNA statistics, so
example windows can be chosen by rank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eqtl"))

from benchmark import _load_model, load_sequence_tools  # noqa: E402

from alphagenome_ft.finetune import load_targets_config, prepare_head_specs  # noqa: E402
from alphagenome_ft.finetune.data import GeneExpressionSupervision  # noqa: E402
from alphagenome_ft.finetune.reliability import double_centered_pearson  # noqa: E402
from alphagenome_ft.inference_bundle import gene_expression, predict_heads  # noqa: E402

WINDOW = 131072


def observed_128bp(handles: list, chrom: str, start: int, end: int) -> np.ndarray:
    return np.stack([np.nan_to_num(h.values(chrom, start, end, numpy=True)).reshape(-1, 128).mean(axis=1) for h in handles], axis=1)


def overlaps(windows: pd.DataFrame, bed: str | None) -> np.ndarray:
    if not bed:
        return np.zeros(len(windows), dtype=bool)
    regions = pd.read_csv(bed, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], comment="#")
    hit = np.zeros(len(windows), dtype=bool)
    for chrom, group in regions.groupby("chrom"):
        mask = (windows.chrom == chrom).to_numpy()
        starts, ends = windows.start.to_numpy()[mask], windows.end.to_numpy()[mask]
        hit[mask] = np.array([((group.start < e) & (group.end > s)).any() for s, e in zip(starts, ends)], dtype=bool)
    return hit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--dataset", default="zemke2023")
    parser.add_argument("--source", default="human")
    parser.add_argument("--window", action="append", default=[], help="chrom:start, 131,072 bp from start")
    parser.add_argument("--score-split", default=None, help="Score every window of this split, e.g. test")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.dataset_config.read_text())
    source = next(s for d in config["datasets"] if d["name"] == args.dataset for s in d["sources"] if s["name"] == args.source)
    targets = load_targets_config(Path(source["targets_config"]))
    specs = {spec.head_id: spec for spec in prepare_head_specs(targets)}
    atac = next(spec for spec in specs.values() if spec.kind == "atac")
    rna = next(spec for spec in specs.values() if spec.kind == "rna_seq")
    organism_index = 1 if source.get("organism") == "MUS_MUSCULUS" else 0
    model = _load_model("example", args.checkpoint, base_checkpoint=args.base_checkpoint, targets_path=Path(source["targets_config"]), init_seq_len=WINDOW, dataset_config=args.dataset_config)
    genome, extractor, encoder = load_sequence_tools(Path(source["fasta"]))
    annotation = GeneExpressionSupervision(rna.gene_supervision_path, rna)
    groups = list(annotation.groups)
    valid = np.asarray(annotation.group_valid, dtype=bool)
    handles = [pyBigWig.open(str(t.path)) for t in atac.tracks]
    tracks = [t.name for t in atac.tracks]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def predict(chrom: str, start: int):
        interval = genome.Interval(chrom, start, start + WINDOW)
        sequence = encoder.encode(extractor.extract(interval).upper()).astype(np.float32)[None]
        outputs = predict_heads(model, sequence, organism_index)
        # The 1 bp ATAC output averaged per 128 bp bin is in the same per-base units as the observed bins.
        predicted_atac = np.asarray(outputs[atac.head_id]["predictions_1bp"][0], dtype=np.float32).reshape(-1, 128, len(tracks)).mean(axis=1)
        observed_atac = observed_128bp(handles, chrom, start, start + WINDOW)
        gene_ids, predicted_cpm = gene_expression(outputs[rna.head_id]["predictions_128bp"][0], annotation, interval)
        observed_cpm = np.stack([annotation.cpm[:, annotation.gene_ids.index(g)] for g in gene_ids]) if gene_ids else np.zeros((0, len(groups)))
        return observed_atac, predicted_atac, gene_ids, observed_cpm, np.asarray(predicted_cpm).reshape(len(gene_ids), len(groups))

    if args.score_split:
        windows = pd.read_csv(source["split_intervals_bed"], sep="\t", header=None, names=["chrom", "start", "end", "split"])
        windows = windows[windows.split == args.score_split].reset_index(drop=True)
        windows["blacklisted"] = overlaps(windows, source.get("exclude_intervals_bed"))
        window_rows, gene_rows = [], []
        for i, (chrom, start, blacklisted) in enumerate(zip(windows.chrom, windows.start, windows.blacklisted)):
            if blacklisted:
                continue
            observed_atac, predicted_atac, gene_ids, observed_cpm, predicted_cpm = predict(chrom, int(start))
            window_rows.append((chrom, int(start), double_centered_pearson(predicted_atac, observed_atac), float(observed_atac.mean()), len(gene_ids)))
            for j, g in enumerate(gene_ids):
                index = annotation.gene_ids.index(g)
                obs, pred = np.log10(1 + observed_cpm[j, valid]), np.log10(1 + predicted_cpm[j, valid])
                r = float(np.corrcoef(obs, pred)[0, 1]) if obs.std() > 0 and pred.std() > 0 else float("nan")
                inside = annotation.starts[index] >= start and annotation.ends[index] <= start + WINDOW
                gene_rows.append((chrom, int(start), g, bool(inside), float(observed_cpm[j, valid].max()), float(obs.std()), r))
            if i % 50 == 0:
                print(f"{i}/{len(windows)}", flush=True)
        stem = args.output_dir / f"{args.dataset}_{args.source}_{args.score_split}"
        pd.DataFrame(window_rows, columns=["chrom", "start", "atac_r", "observed_mean", "n_genes"]).to_csv(f"{stem}_windows.tsv.gz", sep="\t", index=False)
        pd.DataFrame(gene_rows, columns=["chrom", "start", "gene", "inside", "observed_max_cpm", "observed_log_sd", "rna_r"]).to_csv(f"{stem}_genes.tsv.gz", sep="\t", index=False)
        print("skipped blacklisted", int(windows.blacklisted.sum()), "scored", len(window_rows), flush=True)

    for value in args.window:
        chrom, start = value.split(":")
        start = int(start)
        observed_atac, predicted_atac, gene_ids, observed_cpm, predicted_cpm = predict(chrom, start)
        stem = args.output_dir / f"{args.dataset}_{args.source}_{chrom}_{start}"
        rows = [(b * 128 + start, track, float(observed_atac[b, i]), float(predicted_atac[b, i])) for i, track in enumerate(tracks) for b in range(predicted_atac.shape[0])]
        pd.DataFrame(rows, columns=["position", "group", "observed", "predicted"]).to_csv(f"{stem}_atac.tsv.gz", sep="\t", index=False)
        rna_rows = [(g, group, float(observed_cpm[j, k]), float(predicted_cpm[j, k]), bool(valid[k])) for j, g in enumerate(gene_ids) for k, group in enumerate(groups)]
        pd.DataFrame(rna_rows, columns=["gene", "group", "observed_cpm", "predicted_cpm", "valid"]).to_csv(f"{stem}_rna.tsv.gz", sep="\t", index=False)
        exons = annotation.exon_offsets
        gene_rows = []
        for g in gene_ids:
            index = annotation.gene_ids.index(g)
            for exon_start, exon_end in zip(annotation.exon_starts[exons[index]:exons[index + 1]], annotation.exon_ends[exons[index]:exons[index + 1]]):
                gene_rows.append((g, int(annotation.starts[index]), int(annotation.ends[index]), annotation.strands[index], int(exon_start), int(exon_end)))
        pd.DataFrame(gene_rows, columns=["gene", "start", "end", "strand", "exon_start", "exon_end"]).to_csv(f"{stem}_genes.tsv.gz", sep="\t", index=False)
        print(value, "genes", gene_ids, "atac_r", double_centered_pearson(predicted_atac, observed_atac), flush=True)
    for handle in handles:
        handle.close()


if __name__ == "__main__":
    main()
