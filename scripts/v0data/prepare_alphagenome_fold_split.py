"""Rewrite a v0data dataset config to split windows by AlphaGenome pretraining folds.

Human and mouse windows are labeled directly from the checkpoint's held-out
fold regions. Macaque and marmoset windows are labeled by lifting 1 kb bins to
hg38, so that sequence orthologous to held-out human regions is held out too.
Windows that mix splits are dropped. Validation and test windows are then
subsampled, with a fixed seed, to the number of windows each source had on its
original validation and test chromosomes, so evaluation cost and precision
match the chromosome split. Each source gains a ``split_intervals_bed`` that
the training script uses in place of its chromosome split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome_ft.finetune.fold_split import (
    alphagenome_fold_regions,
    label_windows_direct,
    label_windows_lifted,
    liftover_bins,
    tile_windows,
)


def _chrom_set(value: str) -> set[str]:
    return {item for item in str(value or "").split(",") if item}


def _fai_sizes(fasta: str) -> dict[str, int]:
    fai = pd.read_csv(f"{fasta}.fai", sep="\t", header=None, usecols=[0, 1], names=["chrom", "size"])
    return dict(zip(fai.chrom, fai["size"]))


def _routes(chains_dir: Path) -> dict[str, dict]:
    """Liftover route to hg38 for each non-human, non-mouse training FASTA basename."""
    rhe = pd.read_csv(chains_dir / "rheMac10.chromAlias.txt", sep="\t", header=None, comment="#", names=["ucsc", "genbank", "refseq"])
    marmoset = pd.read_csv(chains_dir / "GCA_011100555.2.chromAlias.txt", sep="\t")
    marmoset.columns = [column.lstrip("# ") for column in marmoset.columns]
    return {
        "rheMac10.fa": {"chains": [chains_dir / "rheMac10ToHg38.over.chain.gz"], "min_match": [0.5]},
        "calJac4.fa": {"chains": [chains_dir / "calJac4ToHg38.over.chain.gz"], "min_match": [0.5]},
        # Johansen macaque uses RefSeq accessions for rheMac10 (Mmul_10) chromosomes.
        "macaque.fa": {"chains": [chains_dir / "rheMac10ToHg38.over.chain.gz"], "min_match": [0.5], "rename": dict(zip(rhe.refseq, rhe.ucsc))},
        # Johansen marmoset (mCalJa1.2.pat.X) names chromosomes chr<NCBI name>; lift via calJac4.
        "marmoset.fa": {
            "chains": [chains_dir / "GCA_011100555.2ToCalJac4.over.chain.gz", chains_dir / "calJac4ToHg38.over.chain.gz"],
            "min_match": [0.95, 0.5],
            "rename": dict(zip("chr" + marmoset.ncbi.astype(str), marmoset.genbank)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chains-dir", type=Path, required=True)
    parser.add_argument("--model-version", default="FOLD_1")
    parser.add_argument("--window-size", type=int, default=131072)
    parser.add_argument("--stride", type=int, default=131072)
    parser.add_argument("--liftover", default="liftOver")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--reuse-labels", action="store_true", help="Read labels from existing splits/full BEDs")
    parser.add_argument("--bin-size", type=int, default=4000, help="Liftover bin size for non-human, non-mouse sources")
    parser.add_argument("--sources", default="", help="Comma-separated dataset:source routes to label only (no config written)")
    args = parser.parse_args()

    config = json.loads(args.input.read_text())
    regions = {organism: alphagenome_fold_regions(args.model_version, organism) for organism in ("HOMO_SAPIENS", "MUS_MUSCULUS")}
    routes = _routes(args.chains_dir)
    split_dir = args.output_dir / "splits"
    full_dir = split_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    cache: dict[tuple, pd.Series] = {}
    only = set(filter(None, args.sources.split(",")))
    for dataset in config["datasets"]:
        for source in dataset["sources"]:
            if only and f"{dataset['name']}:{source['name']}" not in only:
                continue
            fasta = source["fasta"]
            windows = tile_windows(
                _fai_sizes(fasta),
                window_size=args.window_size,
                stride=args.stride,
                include_chroms=_chrom_set(source.get("include_chroms")) or None,
                exclude_chroms=_chrom_set(source.get("exclude_chroms")),
            )
            key = (fasta, source.get("include_chroms", ""), source.get("exclude_chroms", ""))
            full_bed = full_dir / f"{dataset['name']}_{source['name']}.bed"
            if args.reuse_labels and full_bed.exists() and key not in cache:
                previous = pd.read_csv(full_bed, sep="\t", header=None, names=["chrom", "start", "end", "split"])
                labels = windows.merge(previous, on=["chrom", "start", "end"], how="left").split
                cache[key] = labels.where(labels.notna(), None).set_axis(windows.index)
            if key not in cache:
                name = Path(fasta).name
                if name == "hg38.fa":
                    cache[key] = label_windows_direct(windows, regions["HOMO_SAPIENS"])
                elif name == "mm10.fa":
                    cache[key] = label_windows_direct(windows, regions["MUS_MUSCULUS"])
                else:
                    route = routes[name]
                    lifted = liftover_bins(windows, route["chains"], bin_size=args.bin_size, min_match=route["min_match"], rename=route.get("rename"), liftover=args.liftover)
                    cache[key] = label_windows_lifted(windows, lifted, regions["HOMO_SAPIENS"], bin_size=args.bin_size)
            labels = cache[key]
            kept = windows.assign(split=labels.to_numpy()).dropna(subset=["split"])
            kept.to_csv(full_bed, sep="\t", header=False, index=False)
            rng = np.random.default_rng([args.seed, len(summary)])
            targets = {
                "valid": int(windows.chrom.isin(_chrom_set(source["valid_chroms"])).sum()),
                "test": int(windows.chrom.isin(_chrom_set(source["test_chroms"])).sum()),
            }
            for split, target in targets.items():
                rows = np.flatnonzero((kept.split == split).to_numpy())
                if len(rows) > target:
                    drop = rng.choice(rows, size=len(rows) - target, replace=False)
                    kept = kept.drop(kept.index[drop])
            bed = split_dir / f"{dataset['name']}_{source['name']}.bed"
            kept.sort_values(["chrom", "start"]).to_csv(bed, sep="\t", header=False, index=False)
            source["split_intervals_bed"] = str(bed.resolve())
            source["valid_chroms"] = ""
            source["test_chroms"] = ""
            counts = kept.split.value_counts().to_dict()
            summary[f"{dataset['name']}:{source['name']}"] = {
                "windows": len(windows),
                "dropped": int(labels.isna().sum()),
                "available_valid": int((labels == "valid").sum()),
                "available_test": int((labels == "test").sum()),
                **{split: int(counts.get(split, 0)) for split in ("train", "valid", "test")},
            }
            print(dataset["name"], source["name"], summary[f"{dataset['name']}:{source['name']}"], flush=True)
    if only:
        return
    config["split_variant"] = {
        "name": "alphagenome_fold",
        "model_version": args.model_version,
        "window_size": args.window_size,
        "stride": args.stride,
        "heldout_subsample": "valid/test sampled to the original chromosome-split window counts",
        "rule": f"direct fold overlap for hg38/mm10 (no other-split bases); {args.bin_size} bp liftover bins to hg38 otherwise (>=20% bins held out, <=5% other)",
    }
    (args.output_dir / "datasets.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
