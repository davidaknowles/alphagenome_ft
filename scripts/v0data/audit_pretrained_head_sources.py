#!/usr/bin/env python3
"""Audit pretrained ATAC and RNA source channels used by neural bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib

from alphagenome_ft.custom_model import _neural_source_candidate_mask


OUTPUT_TYPES = (dna_output.OutputType.ATAC, dna_output.OutputType.RNA_SEQ)


def audit_organism(organism: dna_model.Organism) -> dict[str, Any]:
    metadata = metadata_lib.load(organism)
    assays = []
    for output_type in OUTPUT_TYPES:
        frame = metadata.get(output_type)
        valid = tuple(bool(value) for value in ~metadata.padding[output_type])
        neural = _neural_source_candidate_mask(frame, source_valid=valid)
        strands = tuple(frame["strand"].astype(str))
        labels = frame["biosample_name"].fillna("").astype(str)
        assays.append(
            {
                "assay": output_type.name.lower(),
                "valid_channels": sum(valid),
                "neural_channels": sum(neural),
                "valid_by_strand": {
                    strand: sum(keep and value == strand for keep, value in zip(valid, strands))
                    for strand in sorted(set(strands))
                },
                "neural_by_strand": {
                    strand: sum(keep and value == strand for keep, value in zip(neural, strands))
                    for strand in sorted(set(strands))
                },
                "neural_labels": [
                    label for label, keep in zip(labels, neural, strict=True) if keep
                ],
            }
        )
    return {"organism": organism.name, "assays": assays}


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Pretrained neural-head source audit",
        "",
        "The neural bootstrap uses a neural source pool only when every target strand has at least two eligible pretrained channels. Otherwise it retains the complete assay-wide pool.",
        "",
        "| Organism | Assay | Valid channels | Neural channels | Valid by strand | Neural by strand |",
        "|---|---|---:|---:|---|---|",
    ]
    for organism in result["organisms"]:
        for assay in organism["assays"]:
            valid = ", ".join(
                f"{strand}={count}" for strand, count in assay["valid_by_strand"].items()
            )
            neural = ", ".join(
                f"{strand}={count}" for strand, count in assay["neural_by_strand"].items()
            )
            lines.append(
                f"| `{organism['organism']}` | `{assay['assay']}` | "
                f"{assay['valid_channels']} | {assay['neural_channels']} | "
                f"{valid} | {neural} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    result = {"organisms": [audit_organism(organism) for organism in dna_model.Organism]}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(result))
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
