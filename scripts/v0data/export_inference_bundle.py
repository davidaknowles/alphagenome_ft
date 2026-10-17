"""Export a selected v0data checkpoint as a portable inference bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from alphagenome_ft.inference_bundle import export_inference_bundle, write_checksums

REPO_ROOT = Path(__file__).resolve().parents[2]

# Assemblies behind the training FASTA basenames in the v0data dataset configs.
REFERENCE_ASSEMBLIES = {
    "hg38.fa": "hg38 (GRCh38), UCSC chromosome names",
    "macaque.fa": "Mmul_10 (GCF_003339765.1), RefSeq accession names",
    "marmoset.fa": "mCalJa1.2.pat.X (GCA_011100555.2), chr-style names",
    "rheMac10.fa": "rheMac10 (UCSC)",
    "calJac4.fa": "calJac4 (UCSC)",
    "mm10.fa": "mm10 (UCSC)",
}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Selected checkpoint, e.g. <run>/best.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, help="Native-source evaluation JSON for the run.")
    parser.add_argument(
        "--docs",
        type=Path,
        default=REPO_ROOT / "docs/v0data_inference_bundle",
        help="Directory of README.md/AGENTS.md copied into the bundle.",
    )
    args = parser.parse_args()

    run_dir = args.checkpoint.resolve().parent
    continuation = json.loads((run_dir / "continuation.json").read_text())
    metrics = json.loads((args.checkpoint / "metrics.json").read_text())
    provenance = {
        "run": run_dir.name,
        "selected_checkpoint": args.checkpoint.name,
        "epoch": metrics["epoch"],
        "global_step": metrics["global_step"],
        "adapters": {key: continuation[key] for key in continuation if key.startswith(("lora_", "locon_"))},
        "head_warmup_checkpoint": Path(continuation["source_checkpoint"]).name,
        "dataset_config": Path(continuation["dataset_config"]).parent.name,
        "alphagenome_ft_commit": _git_commit(),
        "base_model": "AlphaGenome JAX all_folds",
        "reference_assemblies": REFERENCE_ASSEMBLIES,
    }
    if args.evaluation is not None:
        provenance["evaluation"] = args.evaluation.stem
    bundle = export_inference_bundle(
        args.checkpoint, REPO_ROOT / continuation["dataset_config"], args.output, provenance=provenance
    )
    if args.evaluation is not None:
        (bundle / "evaluation.md").write_text(args.evaluation.with_suffix(".md").read_text())
    for name in ("README.md", "AGENTS.md"):
        if (args.docs / name).exists():
            (bundle / name).write_text((args.docs / name).read_text())
    write_checksums(bundle)
    print(f"wrote {bundle}")


if __name__ == "__main__":
    main()
