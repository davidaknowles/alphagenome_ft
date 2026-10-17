"""Check that an inference bundle reproduces predictions from the original checkpoint.

Run ``--mode original`` and ``--mode bundle`` in separate processes (head
registration is global), then ``--mode compare --names original bundle``.
Repeating ``original`` under another ``--name`` gives the run-to-run baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = Path.home() / ".cache/kagglehub/models/google/alphagenome/jax/all_folds/1"
# A 131,072 bp hg38 window containing APOE, TOMM40 and APOC1.
WINDOW = ("chr19", 44_840_000, 44_840_000 + 2**17)


def _sequence(fasta: Path) -> np.ndarray:
    from alphagenome.data import genome
    from alphagenome_research.io import fasta as fasta_lib
    from alphagenome_research.model import one_hot_encoder

    interval = genome.Interval(*WINDOW)
    sequence = fasta_lib.FastaExtractor(str(fasta)).extract(interval).upper()
    return one_hot_encoder.DNAOneHotEncoder(dtype=np.float32).encode(sequence)[None]


def _load_original(checkpoint: Path, base: Path):
    from alphagenome_ft import load_checkpoint
    from alphagenome_ft.finetune import load_targets_config, prepare_head_specs, register_predefined_heads

    run_dir = checkpoint.resolve().parent
    continuation = json.loads((run_dir / "continuation.json").read_text())
    datasets = json.loads((REPO_ROOT / continuation["dataset_config"]).read_text())["datasets"]
    specs = [
        spec
        for dataset in datasets
        for source in dataset["sources"]
        for spec in prepare_head_specs(load_targets_config(Path(source["targets_config"])))
    ]
    register_predefined_heads(specs)
    return load_checkpoint(checkpoint, base_checkpoint_path=str(base), init_seq_len=WINDOW[2] - WINDOW[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("original", "bundle", "compare"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--fasta", type=Path, default=Path("/gpfs/commons/groups/knowles_lab/index/hg38/hg38.fa"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", help="Output name for original/bundle modes; defaults to the mode.")
    parser.add_argument("--names", nargs=2, default=("original", "bundle"), help="Two outputs to compare.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        first, second = (np.load(args.output_dir / f"{name}.npz") for name in args.names)
        assert sorted(first.files) == sorted(second.files), (first.files, second.files)
        report = {}
        for key in sorted(first.files):
            a, b = first[key], second[key]
            assert a.shape == b.shape, (key, a.shape, b.shape)
            report[key] = {
                "shape": list(a.shape),
                "max_abs_diff": float(np.max(np.abs(a - b))),
                "max_rel_diff": float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-12)),
                "pearson": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
            }
        name = "_vs_".join(args.names)
        (args.output_dir / f"{name}.json").write_text(json.dumps(report, indent=2))
        worst = max(report.values(), key=lambda item: item["max_rel_diff"])
        print(
            f"{name}: {len(report)} arrays, worst max_rel_diff={worst['max_rel_diff']:.3g}, "
            f"min pearson={min(item['pearson'] for item in report.values()):.7f}"
        )
        return

    from alphagenome.data import genome
    from alphagenome_ft.inference_bundle import (
        gene_expression,
        head_organism_index,
        load_gene_annotation,
        load_inference_bundle,
        predict_heads,
        read_bundle_heads,
    )

    if args.mode == "original":
        model = _load_original(args.checkpoint, args.base_checkpoint)
    else:
        model = load_inference_bundle(
            args.bundle, base_checkpoint_path=args.base_checkpoint, init_seq_len=WINDOW[2] - WINDOW[1]
        )
    sequence = _sequence(args.fasta)
    arrays = {}
    for organism_index in (0, 1):
        for head, outputs in predict_heads(model, sequence, organism_index).items():
            for name, value in outputs.items():
                arrays[f"org{organism_index}/{head}/{name}"] = np.asarray(value, dtype=np.float32)
    if args.mode == "bundle":
        heads = {head["id"]: head for head in read_bundle_heads(args.bundle)}
        assert head_organism_index(heads["zemke2023_rna_human"]) == 0
        assert head_organism_index(heads["zemke2023_rna_mouse"]) == 1
        annotation = load_gene_annotation(args.bundle, "zemke2023_rna_human")
        gene_ids, values = gene_expression(
            arrays["org0/zemke2023_rna_human/predictions_128bp"][0], annotation, genome.Interval(*WINDOW)
        )
        labels = [target["label"] for target in heads["zemke2023_rna_human"]["targets"]]
        print("genes in window:", gene_ids)
        print("tracks:", labels)
        print(np.array2string(values, precision=2, max_line_width=200))
    np.savez(args.output_dir / f"{args.name or args.mode}.npz", **arrays)
    print(f"saved {len(arrays)} arrays")


if __name__ == "__main__":
    main()
