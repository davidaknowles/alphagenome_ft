#!/usr/bin/env python3
"""Prepare a matched Zemke 2023 marmoset evaluation excluding ANO6 windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUTLIER_CHROMOSOME = "chr9"
OUTLIER_START = 27_738_607
OUTLIER_END = 27_738_948
OUTLIER_LABEL = "SSU-rRNA_Hsa"


def prepare_audit(
    *,
    dataset_config_path: Path,
    supervision_path: Path,
    output_dir: Path,
    gene_id: str,
) -> dict[str, object]:
    """Write the exclusion BED and a dataset configuration that references it."""
    dataset_config_path = dataset_config_path.resolve()
    supervision_path = supervision_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(supervision_path, allow_pickle=False) as supervision:
        gene_ids = supervision["gene_ids"].astype(str)
        matches = np.flatnonzero(gene_ids == gene_id)
        if len(matches) != 1:
            raise ValueError(f"Expected one {gene_id} gene, found {len(matches)}.")
        index = int(matches[0])
        chromosome = str(supervision["chromosomes"][index])
        start = int(supervision["starts"][index])
        end = int(supervision["ends"][index])
        groups = supervision["groups"].astype(str)
        cpm = np.asarray(supervision["cpm"][:, index], dtype=np.float64)

    if not (
        chromosome == OUTLIER_CHROMOSOME
        and start <= OUTLIER_START < OUTLIER_END <= end
    ):
        raise ValueError("The annotated ribosomal RNA outlier does not lie within ANO6.")
    exclusion_bed = (output_dir / "ssu_rrna_hsa.bed").resolve()
    exclusion_bed.write_text(
        f"{OUTLIER_CHROMOSOME}\t{OUTLIER_START}\t{OUTLIER_END}\t{OUTLIER_LABEL}\n"
    )
    payload = json.loads(dataset_config_path.read_text())
    matched_source = None
    for dataset in payload["datasets"]:
        if dataset["name"] != "zemke2023":
            continue
        for source in dataset["sources"]:
            if source["name"] == "marmoset":
                source["exclude_intervals_bed"] = str(exclusion_bed)
                matched_source = source
    if matched_source is None:
        raise ValueError("Dataset config lacks the zemke2023:marmoset source.")

    output_config = output_dir / "datasets.json"
    output_config.write_text(json.dumps(payload, indent=2) + "\n")
    result = {
        "gene_id": gene_id,
        "chromosome": chromosome,
        "start": start,
        "end": end,
        "span_bp": end - start,
        "maximum_cpm": float(cpm.max()),
        "maximum_cpm_group": str(groups[int(cpm.argmax())]),
        "median_cpm": float(np.median(cpm)),
        "excluded_repeat": {
            "label": OUTLIER_LABEL,
            "chromosome": OUTLIER_CHROMOSOME,
            "start": OUTLIER_START,
            "end": OUTLIER_END,
            "repeat_class": "rRNA",
            "repeat_family": "rRNA",
        },
        "source_dataset_config": str(dataset_config_path),
        "source_gene_supervision": str(supervision_path),
        "excluded_dataset_config": str(output_config.resolve()),
        "exclusion_bed": str(exclusion_bed),
    }
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--supervision", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gene-id", default="ANO6")
    args = parser.parse_args()
    result = prepare_audit(
        dataset_config_path=args.dataset_config,
        supervision_path=args.supervision,
        output_dir=args.output_dir,
        gene_id=args.gene_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
