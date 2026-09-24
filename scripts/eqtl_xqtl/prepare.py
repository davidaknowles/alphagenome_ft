#!/usr/bin/env python
"""Prepare XQTL records and compare fine-mapping with singlebrain PIPs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eqtl.benchmark import (  # noqa: E402
    BASES,
    DEFAULT_FINEMAP_DIR,
    DEFAULT_GTF,
    Gene,
    Record,
    Reservoir,
    _stable_seed,
    load_genes,
    load_records_cache,
    save_records,
)

DEFAULT_XQTL_ROOT = Path(
    "/gpfs/commons/groups/knowles_lab/gcavalca/mlxqtl_analysis/data/"
    "susie_vars_pips_04_2024"
)
XQTL_GROUPS = {
    "Ast": ("Ast", "ASC"),
    "Exc": ("Ext", "EXC"),
    "Inh": ("IN", "INH"),
    "Mic": ("MG", "MGC"),
    "OPC": ("OPC", "OPC"),
    "Oli": ("OD", "ODC"),
}
SINGLEBRAIN_FILES = {
    "Ast": "Ast_all.susie_all_pip.tsv.gz",
    "Ext": "Ext_all.susie_all_pip.tsv.gz",
    "IN": "IN_all.susie_all_pip.tsv.gz",
    "MG": "MG_all.susie_all_pip.tsv.gz",
    "OPC": "OPC_all.susie_all_pip.tsv.gz",
    "OD": "OD_all.susie_all_pip.tsv.gz",
}


def _valid_variant(chrom: str, pos: int, ref: str, alt: str, feature: str,
                   genes: dict[str, Gene]) -> tuple[str, int, str, str, Gene] | None:
    gene = genes.get(feature.split(".", 1)[0])
    ref, alt = ref.upper(), alt.upper()
    if gene is None or gene.chrom != chrom:
        return None
    if len(ref) != 1 or len(alt) != 1 or ref not in BASES or alt not in BASES or ref == alt:
        return None
    pos0 = pos - 1
    if abs(pos0 - gene.tss) >= 1_048_576 // 2:
        return None
    return feature.split(".", 1)[0], pos0, ref, alt, gene


def _scan_xqtl(args, genes):
    records: list[Record] = []
    counts: dict[str, dict[str, int]] = {}
    positive_sets: dict[str, set[tuple[str, str, int, str, str]]] = {}
    negative_reservoirs: dict[str, Reservoir] = {}

    for cell_type, (panel, group) in XQTL_GROUPS.items():
        started = time.monotonic()
        print(f"opening XQTL {cell_type}", flush=True)
        dataset_path = args.xqtl_root / f"{cell_type}_mega_eQTL/PIP_all_parquet/PIP_all.parquet"
        dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
        if not {"pos", "ref", "alt", "pip", "gene_id"}.issubset(dataset.schema.names):
            raise ValueError(f"Unexpected schema in {dataset_path}: {dataset.schema}")
        counts[panel] = {
            "rows": 0, "eligible_rows": 0, "excluded_rows": 0,
            "positive": 0, "negative": 0,
            "retained_positive": 0, "retained_negative": 0,
        }
        positive_sets[panel] = set()
        negative_reservoirs[panel] = Reservoir(
            args.max_negatives_per_group, _stable_seed(str(args.seed), panel, "negative")
        )
        scan_columns = ["pos", "ref", "alt", "pip", "gene_id"]
        has_partition_chrom = "chr" in dataset.schema.names
        if has_partition_chrom:
            scan_columns.append("chr")
        else:
            scan_columns.append("variant_id")
        batch_number = 0
        for batch in dataset.to_batches(
            columns=scan_columns,
            batch_size=65_536,
        ):
            batch_number += 1
            for row in batch.to_pylist():
                counts[panel]["rows"] += 1
                try:
                    chrom = str(row.get("chr") or row["variant_id"].split(":", 1)[0])
                    pos = int(row["pos"])
                    pip = float(row["pip"])
                    feature = str(row["gene_id"])
                    variant = _valid_variant(chrom, pos, str(row["ref"]), str(row["alt"]), feature, genes)
                except (TypeError, ValueError):
                    variant = None
                if variant is None:
                    counts[panel]["excluded_rows"] += 1
                    continue
                feature, pos0, ref, alt, gene = variant
                counts[panel]["eligible_rows"] += 1
                key = (feature, gene.chrom, pos0, ref, alt)
                label = 1 if pip > 0.75 else 0 if pip < 0.01 else None
                if label is None:
                    continue
                counts[panel]["positive" if label else "negative"] += 1
                record = Record(panel, feature, gene.chrom, pos0, ref, alt, pip, label, group, gene)
                if label:
                    positive_sets[panel].add(key)
                    records.append(record)
                else:
                    negative_reservoirs[panel].add(record)
            if batch_number % 50 == 0:
                print(
                    f"scanned XQTL {cell_type}: {counts[panel]['rows']:,} rows "
                    f"in {time.monotonic() - started:.0f}s",
                    flush=True,
                )
        records.extend(negative_reservoirs[panel].items)
        counts[panel]["retained_positive"] = counts[panel]["positive"]
        counts[panel]["retained_negative"] = len(negative_reservoirs[panel].items)
        print(
            f"prepared {cell_type}: rows={counts[panel]['rows']:,}, "
            f"eligible={counts[panel]['eligible_rows']:,}, "
            f"positive={counts[panel]['positive']:,}, negative={counts[panel]['negative']:,}",
            flush=True,
        )
    records.sort(key=lambda r: (r.panel, r.label, r.feature, r.pos, r.alt))
    return records, counts, positive_sets


def _scan_singlebrain(args, genes, positive_sets, counts):
    overlap_rows = []
    for panel, filename in SINGLEBRAIN_FILES.items():
        path = args.singlebrain_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        xqtl_panel = panel
        xqtl_positive = positive_sets[xqtl_panel]
        sb_positive: set[tuple[str, str, int, str, str]] = set()
        eligible = 0
        invalid = 0
        rows_seen = 0
        started = time.monotonic()
        with gzip.open(path, "rt") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                rows_seen += 1
                if rows_seen % 5_000_000 == 0:
                    print(
                        f"scanned singlebrain {panel}: {rows_seen:,} rows "
                        f"in {time.monotonic() - started:.0f}s",
                        flush=True,
                    )
                try:
                    feature = row["feature"].split(".", 1)[0]
                    chrom = row["chr"] if row["chr"].startswith("chr") else f"chr{row['chr']}"
                    pos = int(row["pos"])
                    pip = float(row["susie_pip"])
                    variant = _valid_variant(chrom, pos, row["ref"], row["alt"], feature, genes)
                except (KeyError, TypeError, ValueError):
                    variant = None
                if variant is None:
                    invalid += 1
                    continue
                feature, pos0, ref, alt, gene = variant
                eligible += 1
                key = (feature, gene.chrom, pos0, ref, alt)
                if pip > 0.75:
                    sb_positive.add(key)
        shared_positive = xqtl_positive & sb_positive
        union = len(xqtl_positive | sb_positive)
        overlap_rows.append({
            "panel": panel,
            "xqtl_eligible_rows": counts[panel]["eligible_rows"],
            "xqtl_sampled_rows": len(xqtl_sample),
            "singlebrain_eligible_rows": eligible,
            "singlebrain_invalid_or_outside_window": invalid,
            "xqtl_high_pip": len(xqtl_positive),
            "singlebrain_high_pip": len(sb_positive),
            "shared_high_pip": len(shared_positive),
            "high_pip_jaccard": len(shared_positive) / union if union else float("nan"),
            "xqtl_high_pip_recall": len(shared_positive) / len(xqtl_positive) if xqtl_positive else float("nan"),
            "singlebrain_high_pip_recall": len(shared_positive) / len(sb_positive) if sb_positive else float("nan"),
        })
        print(f"compared {panel}: eligible singlebrain rows={eligible:,}, shared high-PIP calls={len(shared_positive):,}", flush=True)
    return overlap_rows


def refresh_positive_panel(args, genes, panel: str, group: str) -> None:
    """Replace one panel's cached positives without rescanning the full dataset."""
    cell_type = next(cell for cell, mapping in XQTL_GROUPS.items() if mapping == (panel, group))
    dataset_path = args.xqtl_root / f"{cell_type}_mega_eQTL/PIP_all_parquet/PIP_all.parquet"
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
    columns = ["pos", "ref", "alt", "pip", "gene_id"]
    has_partition_chrom = "chr" in dataset.schema.names
    columns.append("chr" if has_partition_chrom else "variant_id")
    positives: list[Record] = []
    seen: set[tuple[str, str, int, str, str]] = set()
    for batch in dataset.to_batches(
        columns=columns,
        filter=ds.field("pip") > 0.75,
        batch_size=65_536,
    ):
        for row in batch.to_pylist():
            try:
                chrom = str(row.get("chr") or row["variant_id"].split(":", 1)[0])
                variant = _valid_variant(
                    chrom, int(row["pos"]), str(row["ref"]), str(row["alt"]),
                    str(row["gene_id"]), genes,
                )
            except (TypeError, ValueError):
                variant = None
            if variant is None:
                continue
            feature, pos, ref, alt, gene = variant
            key = (feature, gene.chrom, pos, ref, alt)
            if key in seen:
                continue
            seen.add(key)
            positives.append(Record(panel, feature, gene.chrom, pos, ref, alt, float(row["pip"]), 1, group, gene))

    cache_path = args.output_dir / "records.jsonl"
    records = [r for r in load_records_cache(cache_path) if not (r.panel == panel and r.label == 1)]
    records.extend(positives)
    records.sort(key=lambda r: (r.panel, r.label, r.feature, r.pos, r.alt))
    save_records(cache_path, records)
    counts_path = args.output_dir / "sampling_counts.json"
    counts = json.loads(counts_path.read_text())
    counts[panel]["retained_positive"] = len(positives)
    counts_path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
    config_path = args.output_dir / "analysis_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config.pop("agreement_sample_size_per_panel", None)
        config.pop("benchmark_max_per_class_per_panel", None)
        config["max_negative_records_per_panel"] = 1000
        config["positive_records"] = "all eligible records with PIP >0.75"
        config["negative_records"] = "up to 1,000 eligible records with PIP <0.01 per panel"
        config["note"] = "Exact binary high-PIP overlap is calculated after common GTF gene, biallelic SNP and +/-512 kb TSS filters."
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"refreshed {panel}: retained all {len(positives):,} eligible positives; cache now has {len(records):,} records")


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xqtl-root", type=Path, default=DEFAULT_XQTL_ROOT)
    parser.add_argument("--singlebrain-dir", type=Path, default=DEFAULT_FINEMAP_DIR)
    parser.add_argument("--gtf", type=Path, default=DEFAULT_GTF)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-negatives-per-group", type=int, default=1000)
    parser.add_argument("--refresh-positive-panel", choices=sorted({p for p, _ in XQTL_GROUPS.values()}))
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    genes = load_genes(args.gtf)
    if args.refresh_positive_panel:
        panel = args.refresh_positive_panel
        group = next(group for candidate_panel, group in XQTL_GROUPS.values() if candidate_panel == panel)
        refresh_positive_panel(args, genes, panel, group)
        return
    records, counts, positives = _scan_xqtl(args, genes)
    save_records(args.output_dir / "records.jsonl", records)
    (args.output_dir / "sampling_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
    overlap_rows = _scan_singlebrain(args, genes, positives, counts)
    _write_tsv(args.output_dir / "finemap_overlap.tsv", overlap_rows)
    (args.output_dir / "analysis_config.json").write_text(json.dumps({
        "xqtl_input": str(args.xqtl_root),
        "singlebrain_input": str(args.singlebrain_dir),
        "gtf": str(args.gtf),
        "window_bp": 1_048_576,
        "positive_pip": ">0.75",
        "negative_pip": "<0.01",
        "max_negative_records_per_panel": args.max_negatives_per_group,
        "panel_mapping": {cell: {"singlebrain_panel": p, "model_group": g} for cell, (p, g) in XQTL_GROUPS.items()},
        "overlap_key": "gene, chromosome, one-based position, ref, alt",
        "note": "XQTL and singlebrain PIPs are compared only after common GTF gene, biallelic SNP and +/-512 kb from gene TSS filters.",
    }, indent=2) + "\n")
    print(f"wrote {len(records)} benchmark records with uncapped positives", flush=True)


if __name__ == "__main__":
    main()
