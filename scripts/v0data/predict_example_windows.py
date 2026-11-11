#!/usr/bin/env python
"""Predict example held-out windows for one source and save observed and predicted signal.

For each window, writes 128 bp observed and predicted coverage for the source's
ATAC head and gene-level observed and predicted CPM for its RNA head (exon-weighted
readout, as in training), for figure panels.
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
from alphagenome_ft.inference_bundle import gene_expression, predict_heads  # noqa: E402


def observed_128bp(paths: list[str], chrom: str, start: int, end: int) -> np.ndarray:
    out = []
    for path in paths:
        handle = pyBigWig.open(path)
        values = np.nan_to_num(handle.values(chrom, start, end, numpy=True))
        handle.close()
        out.append(values.reshape(-1, 128).mean(axis=1))
    return np.stack(out, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--dataset", default="zemke2023")
    parser.add_argument("--source", default="human")
    parser.add_argument("--window", action="append", required=True, help="chrom:start, 131,072 bp from start")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.dataset_config.read_text())
    source = next(s for d in config["datasets"] if d["name"] == args.dataset for s in d["sources"] if s["name"] == args.source)
    targets = load_targets_config(Path(source["targets_config"]))
    specs = {spec.head_id: spec for spec in prepare_head_specs(targets)}
    atac = next(spec for spec in specs.values() if spec.kind == "atac")
    rna = next(spec for spec in specs.values() if spec.kind == "rna_seq")
    organism_index = 1 if source.get("organism") == "MUS_MUSCULUS" else 0
    model = _load_model("example", args.checkpoint, base_checkpoint=args.base_checkpoint, targets_path=Path(source["targets_config"]), init_seq_len=131072, dataset_config=args.dataset_config)
    genome, extractor, encoder = load_sequence_tools(Path(source["fasta"]))
    annotation = GeneExpressionSupervision(rna.gene_supervision_path, rna)
    groups = list(annotation.groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for value in args.window:
        chrom, start = value.split(":")
        start = int(start)
        interval = genome.Interval(chrom, start, start + 131072)
        sequence = encoder.encode(extractor.extract(interval).upper()).astype(np.float32)[None]
        outputs = predict_heads(model, sequence, organism_index)
        # The 1 bp ATAC output averaged per 128 bp bin is in the same per-base units as the observed bins.
        predicted_atac = np.asarray(outputs[atac.head_id]["predictions_1bp"][0], dtype=np.float32).reshape(-1, 128, len(atac.tracks)).mean(axis=1)
        observed_atac = observed_128bp([str(t.path) for t in atac.tracks], chrom, start, start + 131072)
        gene_ids, predicted_cpm = gene_expression(outputs[rna.head_id]["predictions_128bp"][0], annotation, interval)
        observed_cpm = np.stack([annotation.cpm[:, annotation.gene_ids.index(g)] for g in gene_ids]) if gene_ids else np.zeros((0, len(groups)))
        stem = args.output_dir / f"{args.dataset}_{args.source}_{chrom}_{start}"
        tracks = [t.name for t in atac.tracks]
        rows = [(b * 128 + start, track, float(observed_atac[b, i]), float(predicted_atac[b, i])) for i, track in enumerate(tracks) for b in range(predicted_atac.shape[0])]
        pd.DataFrame(rows, columns=["position", "group", "observed", "predicted"]).to_csv(f"{stem}_atac.tsv.gz", sep="\t", index=False)
        rna_rows = [(g, group, float(observed_cpm[j, k]), float(predicted_cpm[j, k]), bool(annotation.group_valid[k])) for j, g in enumerate(gene_ids) for k, group in enumerate(groups)]
        pd.DataFrame(rna_rows, columns=["gene", "group", "observed_cpm", "predicted_cpm", "valid"]).to_csv(f"{stem}_rna.tsv.gz", sep="\t", index=False)
        exons = annotation.exon_offsets
        gene_rows = []
        for g in gene_ids:
            index = annotation.gene_ids.index(g)
            for exon_start, exon_end in zip(annotation.exon_starts[exons[index]:exons[index + 1]], annotation.exon_ends[exons[index]:exons[index + 1]]):
                gene_rows.append((g, int(annotation.starts[index]), int(annotation.ends[index]), annotation.strands[index], int(exon_start), int(exon_end)))
        pd.DataFrame(gene_rows, columns=["gene", "start", "end", "strand", "exon_start", "exon_end"]).to_csv(f"{stem}_genes.tsv.gz", sep="\t", index=False)
        print(value, "genes", gene_ids, flush=True)


if __name__ == "__main__":
    main()
