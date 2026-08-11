#!/usr/bin/env python
"""Build comparable ATAC and RNA target manifests for v0data experiments."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import bigwig_nonzero_mean, build_head_config

DEFAULT_GRR_ROOT = Path(
    "/gpfs/commons/datasets/controlled/NYGC_AI_Initiative/GRRs/SC_Summaries_GRR/summary"
)
DEFAULT_HDA_ROOT = Path("/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev")
ZEMKE2023_SPECIES = {
    "zemke2023-human": "human_m1",
    "zemke2023-macaque": "macaque_m1",
    "zemke2023-marmoset": "marmoset_m1",
    "zemke2023-mouse": "mouse_mop",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "hda",
            *ZEMKE2023_SPECIES,
            "zemke2024-all",
            "zemke2024-all-ages",
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grr-root", type=Path, default=DEFAULT_GRR_ROOT)
    parser.add_argument("--hda-root", type=Path, default=DEFAULT_HDA_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _discover(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    if args.dataset == "hda":
        atac = sorted((args.hda_root / "bigwigs").glob("*.bw"))
        return atac, []
    if args.dataset in ZEMKE2023_SPECIES:
        root = args.grr_root / "zemke2023Conserved"
        species_dir = ZEMKE2023_SPECIES[args.dataset]
        atac = sorted((root / "pseudo_bulk_atac_bw" / species_dir).glob("*/*.bw"))
        rna = sorted((root / "pseudo_bulk_rna_bw" / species_dir).glob("*/*.bw"))
        return atac, rna
    root = args.grr_root / "zemke2024Epigenetic"
    pattern = "*/all/bigwig/*.bw" if args.dataset == "zemke2024-all" else "*/*/bigwig/*.bw"
    return sorted((root / "ATAC").glob(pattern)), sorted((root / "RNA").glob(pattern))


def _label(path: Path, dataset: str) -> str:
    if dataset in ZEMKE2023_SPECIES:
        return path.parent.name
    if dataset.startswith("zemke2024"):
        cell_type = path.parents[2].name
        age = path.parents[1].name
        return f"{cell_type}_{age}"
    return path.stem


def main() -> None:
    args = _parse_args()
    atac, rna = _discover(args)
    if not atac:
        raise FileNotFoundError(f"No ATAC BigWigs found for {args.dataset}.")
    if args.dataset != "hda" and not rna:
        raise FileNotFoundError(f"No RNA BigWigs found for {args.dataset}.")

    head_prefix = (
        "zemke2023" if args.dataset in ZEMKE2023_SPECIES else args.dataset.replace("-", "_")
    )
    heads = [
        build_head_config(
            head_id=f"{head_prefix}_atac",
            kind="atac",
            tracks=atac,
            labels=[_label(path, args.dataset) for path in atac],
        )
    ]
    if rna:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            nonzero_means = list(executor.map(bigwig_nonzero_mean, rna))
        heads.append(
            build_head_config(
                head_id=f"{head_prefix}_rna",
                kind="rna_seq",
                tracks=rna,
                labels=[_label(path, args.dataset) for path in rna],
                nonzero_means=nonzero_means,
            )
        )

    payload = {
        "dataset": args.dataset,
        "target_contract": {
            "atac": "published_native_signal",
            "rna": "published_native_signal" if rna else None,
            "rna_nonzero_mean": "base_weighted_finite_positive_values" if rna else None,
        },
        "heads": heads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(atac)} ATAC and {len(rna)} RNA targets to {args.output}")


if __name__ == "__main__":
    main()
