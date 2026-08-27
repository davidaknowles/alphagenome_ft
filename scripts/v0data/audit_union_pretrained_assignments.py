#!/usr/bin/env python3
"""Audit deterministic pretrained-head assignments for the joint study panel."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any
import zlib

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib

from alphagenome_ft.custom_model import (
    _bootstrap_track_indices,
    _neural_source_valid,
    _neural_target_channels,
    _semantic_bootstrap_track_indices,
)


KIND_TO_OUTPUT = {
    "atac": dna_output.OutputType.ATAC,
    "rna_seq": dna_output.OutputType.RNA_SEQ,
}


def _organism(source: dict[str, Any]) -> dna_model.Organism:
    name = source.get("organism")
    if name is None:
        name = "MUS_MUSCULUS" if source.get("name") == "mouse" else "HOMO_SAPIENS"
    return dna_model.Organism[str(name)]


def _source_label(frame: pd.DataFrame, index: int) -> str:
    values = []
    for column in ("biosample_name", "gtex_tissue", "name"):
        if column in frame:
            value = str(frame.iloc[index][column])
            if value and value != "nan" and value not in values:
                values.append(value)
    return " | ".join(values)


def audit(config_path: Path, *, semantic: bool = False) -> dict[str, Any]:
    payload = json.loads(config_path.read_text())
    pretrained = {organism: metadata_lib.load(organism) for organism in dna_model.Organism}
    routes = []
    for dataset in payload["datasets"]:
        for source in dataset["sources"]:
            organism = _organism(source)
            targets_path = Path(source["targets_config"])
            targets_payload = json.loads(targets_path.read_text())
            for head in targets_payload["heads"]:
                output_type = KIND_TO_OUTPUT.get(str(head.get("kind")))
                if output_type is None:
                    continue
                source_output_type = (
                    dna_output.OutputType.DNASE
                    if output_type == dna_output.OutputType.ATAC
                    else output_type
                )
                source_frame = pretrained[organism].get(source_output_type)
                source_valid = tuple(
                    bool(value) for value in ~pretrained[organism].padding[source_output_type]
                )
                target_frame = pd.DataFrame(
                    {
                        "name": [str(target["label"]) for target in head["targets"]],
                        "strand": [str(target.get("strand", ".")) for target in head["targets"]],
                    }
                )
                target_strands = tuple(target_frame["strand"])
                neural_targets = _neural_target_channels(target_frame)
                neural_valid = _neural_source_valid(
                    source_frame,
                    source_valid=source_valid,
                    target_strands=target_strands,
                )
                source_valid_by_target = tuple(
                    neural_valid if is_neural else source_valid for is_neural in neural_targets
                )
                seed = zlib.crc32(f"{head['id']}:{organism.name}".encode("utf-8"))
                assignments = (
                    _semantic_bootstrap_track_indices(
                        source_frame,
                        target_frame,
                        source_valid=source_valid,
                        source_valid_by_target=source_valid_by_target,
                        seed=seed,
                    )
                    if semantic
                    else _bootstrap_track_indices(
                        tuple(source_frame["strand"].astype(str)),
                        target_strands,
                        source_valid=source_valid,
                        source_valid_by_target=source_valid_by_target,
                        seed=seed,
                    )
                )
                rows = [
                    {
                        "target_index": target_index,
                        "target_label": str(target_frame.iloc[target_index]["name"]),
                        "target_strand": target_strands[target_index],
                        "neural_target": bool(neural_targets[target_index]),
                        "source_index": int(source_index),
                        "source_label": _source_label(source_frame, source_index),
                        "source_strand": str(source_frame.iloc[source_index]["strand"]),
                    }
                    for target_index, source_index in enumerate(assignments)
                ]
                source_counts = Counter(row["source_index"] for row in rows)
                routes.append(
                    {
                        "dataset": str(dataset["name"]),
                        "source": str(source["name"]),
                        "organism": organism.name,
                        "head": str(head["id"]),
                        "target_kind": str(head["kind"]),
                        "pretrained_assay": source_output_type.name.lower(),
                        "target_channels": len(rows),
                        "neural_target_channels": sum(row["neural_target"] for row in rows),
                        "unique_source_channels": len(source_counts),
                        "maximum_source_reuse": max(source_counts.values(), default=0),
                        "assignments": rows,
                    }
                )
    return {
        "config": str(config_path),
        "initializer": (
            "semantic_neural_accessibility_bootstrap"
            if semantic
            else "neural_accessibility_bootstrap"
        ),
        "routes": routes,
    }


def render_markdown(result: dict[str, Any]) -> str:
    initializer = result.get("initializer", "unknown")
    description = (
        "The semantic variant prefers matching anatomy and cell-class concepts while preserving "
        "assay, strand, and neural-status eligibility. Unmatched targets retain the deterministic "
        "shuffled assignment."
        if initializer == "semantic_neural_accessibility_bootstrap"
        else "The baseline variant deterministically shuffles within each eligible assay, strand, "
        "and neural-status pool without matching target labels to source biosamples."
    )
    lines = [
        "# Joint pretrained-head assignment audit",
        "",
        f"This audit reports the exact `{initializer}` initialization map. {description}",
        "",
        "| Dataset | Native source | Head | Source assay | Targets | Neural targets | "
        "Unique sources | Maximum reuse |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for route in result["routes"]:
        lines.append(
            f"| `{route['dataset']}` | `{route['source']}` | `{route['head']}` | "
            f"`{route['pretrained_assay']}` | {route['target_channels']} | "
            f"{route['neural_target_channels']} | {route['unique_source_channels']} | "
            f"{route['maximum_source_reuse']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    result = audit(args.config.resolve(), semantic=args.semantic)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(result))
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
