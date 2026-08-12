#!/usr/bin/env python
"""Filter paired ATAC and RNA targets by ATAC pseudobulk fragment depth."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.allen_atac_reprocessing.prepare_multispecies_targets import (
    filter_gene_supervision,
)


def filter_target_groups(
    config: dict[str, Any],
    depth_by_group: dict[str, int],
    *,
    minimum_fragments: int,
    atac_head_id: str,
    rna_head_id: str,
) -> tuple[dict[str, Any], list[str]]:
    if minimum_fragments < 0:
        raise ValueError("minimum_fragments must be non-negative.")
    result = copy.deepcopy(config)
    atac_matches = [head for head in result.get("heads", ()) if head.get("id") == atac_head_id]
    rna_matches = [head for head in result.get("heads", ()) if head.get("id") == rna_head_id]
    if len(atac_matches) != 1 or len(rna_matches) != 1:
        raise ValueError("Expected exactly one requested ATAC head and one requested RNA head.")
    atac_head = atac_matches[0]
    rna_head = rna_matches[0]
    groups = [str(target["label"]) for target in atac_head["targets"]]
    missing_depths = [group for group in groups if group not in depth_by_group]
    if missing_depths:
        raise ValueError(f"Fragment depths are missing for groups {missing_depths}.")
    retained = [
        group for group in groups if depth_by_group[group] >= minimum_fragments
    ]
    if not retained:
        raise ValueError("Fragment-depth filtering removed every target group.")
    retained_set = set(retained)
    atac_head["targets"] = [
        target for target in atac_head["targets"] if target["label"] in retained_set
    ]
    rna_by_label = {str(target["label"]): target for target in rna_head["targets"]}
    expected_rna_labels = [
        f"{group} ({strand})" for group in retained for strand in ("+", "-")
    ]
    missing_rna = [label for label in expected_rna_labels if label not in rna_by_label]
    if missing_rna:
        raise ValueError(f"RNA targets are missing paired labels {missing_rna}.")
    rna_head["targets"] = [rna_by_label[label] for label in expected_rna_labels]
    excluded = [group for group in groups if group not in retained_set]
    result["group_depth_filter"] = {
        "minimum_fragments": minimum_fragments,
        "retained_groups": len(retained),
        "excluded_groups": {
            group: int(depth_by_group[group]) for group in excluded
        },
    }
    return result, retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--depths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gene-output", type=Path, required=True)
    parser.add_argument("--minimum-fragments", type=int, required=True)
    parser.add_argument("--atac-head", required=True)
    parser.add_argument("--rna-head", required=True)
    args = parser.parse_args()

    config = json.loads(args.input.read_text())
    with np.load(args.depths, allow_pickle=False) as depth_file:
        depth_by_group = dict(
            zip(
                depth_file["groups"].astype(str).tolist(),
                depth_file["total_fragments"].astype(np.int64).tolist(),
                strict=True,
            )
        )
    filtered, retained = filter_target_groups(
        config,
        depth_by_group,
        minimum_fragments=args.minimum_fragments,
        atac_head_id=args.atac_head,
        rna_head_id=args.rna_head,
    )
    rna_head = next(head for head in filtered["heads"] if head["id"] == args.rna_head)
    gene_supervision = rna_head.get("gene_supervision")
    if not isinstance(gene_supervision, dict) or "path" not in gene_supervision:
        raise ValueError(f"RNA head {args.rna_head!r} lacks gene supervision.")
    filter_gene_supervision(
        Path(gene_supervision["path"]),
        args.gene_output,
        retained,
    )
    gene_supervision["path"] = str(args.gene_output.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(filtered, indent=2) + "\n")
    print(
        f"Retained {len(retained)} groups at >= {args.minimum_fragments:,} fragments in "
        f"{args.output}."
    )


if __name__ == "__main__":
    main()
