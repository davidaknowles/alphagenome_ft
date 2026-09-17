#!/usr/bin/env python
"""Benchmark AlphaGenome mutation effects against fine-mapped single-cell eQTLs.

The benchmark uses the human Zemke 2023 RNA head because its 20 cell labels are
the closest available match to the 32 fine-mapping panels. Fine-mapping panels
are reduced to the matching broad model cell group when a one-to-one label is
not available.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FINEMAP_DIR = Path(
    "/gpfs/commons/groups/knowles_lab/atokolyi/public/singlebrain_full_finemap"
)
DEFAULT_GTF = Path("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/gencode.v38.basic.gtf.gz")
DEFAULT_FASTA = Path("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa")
DEFAULT_BASE = Path("/gpfs/commons/home/daknowles/.cache/kagglehub/models/google/alphagenome/jax/all_folds/1")
DEFAULT_TARGETS = REPO_ROOT / "outputs/v0data/joint-objective-variants/metric-tempered-all-gene-source-specific/zemke2023/human/targets.json"
DEFAULT_CHECKPOINTS = {
    "lora": REPO_ROOT / "checkpoints/v0data/joint_all_nonencode_lora_all_gene_source_specific_semantic_neural_accessibility_bootstrap_headwarm12_adapters_lr3e4_reset/best",
    "locon": REPO_ROOT / "checkpoints/v0data/joint_all_nonencode_lora_locon_all_gene_source_specific_semantic_neural_accessibility_bootstrap_headwarm12_adapters_lr3e4_reset/best",
}

WINDOW = 1_048_576
BIN_EDGES = (0, 1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
BIN_LABELS = ("0-1kb", "1-10kb", "10-50kb", "50-100kb", "100-250kb", "250-500kb", ">500kb")
BASES = frozenset("ACGT")


@dataclass(frozen=True)
class Gene:
    chrom: str
    start: int
    end: int
    strand: str

    @property
    def tss(self) -> int:
        return self.start if self.strand == "+" else self.end - 1


@dataclass(frozen=True)
class Record:
    panel: str
    feature: str
    chrom: str
    pos: int
    ref: str
    alt: str
    pip: float
    label: int
    group: str
    gene: Gene

    @property
    def distance(self) -> int:
        return abs(self.pos - self.gene.tss)

    @property
    def distance_bin(self) -> str:
        for upper, label in zip(BIN_EDGES[1:], BIN_LABELS):
            if self.distance < upper:
                return label
        return BIN_LABELS[-1]


class Reservoir:
    """Bounded deterministic reservoir for one cell panel and class."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.items: list[Record] = []
        self.seen = 0

    def add(self, item: Record) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        index = self.rng.randrange(self.seen)
        if index < self.capacity:
            self.items[index] = item


def _panel_group(panel: str) -> str:
    prefix = re.match(r"[A-Za-z]+", panel).group(0)
    if prefix == "Ast":
        return "ASC"
    if prefix == "End":
        return "Endo"
    if prefix == "Ext":
        return "EXC"
    if prefix == "IN":
        return "INH"
    if prefix == "MG":
        return "MGC"
    if prefix == "OD":
        return "ODC"
    if prefix == "OPC":
        return "OPC"
    raise ValueError(f"Unsupported fine-map panel {panel!r}")


def _track_indices(groups: np.ndarray) -> dict[str, np.ndarray]:
    names = [str(x) for x in groups]
    exact = {name: index for index, name in enumerate(names)}
    return {
        "ASC": np.asarray([exact["ASC"]]),
        "Endo": np.asarray([exact["Endo"]]),
        "EXC": np.asarray([exact[x] for x in ("L2_3_IT", "L4_5_IT", "L5_6_NP", "L5_ET", "L5_IT", "L6_CT", "L6_IT_CAR3", "L6b")]),
        "INH": np.asarray([exact[x] for x in ("LAMP5", "PVALB", "SNCG", "SST", "VIP")]),
        "MGC": np.asarray([exact["MGC"]]),
        "ODC": np.asarray([exact["ODC"]]),
        "OPC": np.asarray([exact["OPC"]]),
    }


def _parse_gtf_attributes(value: str) -> dict[str, str]:
    return {key: val for key, val in re.findall(r'(\w+) "([^"]+)"', value)}


def load_genes(path: Path) -> dict[str, Gene]:
    genes: dict[str, Gene] = {}
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in {"gene", "transcript"}:
                continue
            attrs = _parse_gtf_attributes(fields[8])
            gene_id = attrs.get("gene_id", "").split(".", 1)[0]
            if not gene_id:
                continue
            candidate = Gene(fields[0], int(fields[3]) - 1, int(fields[4]), fields[6])
            previous = genes.get(gene_id)
            if previous is None:
                genes[gene_id] = candidate
            elif previous.chrom == candidate.chrom and previous.strand == candidate.strand:
                genes[gene_id] = Gene(previous.chrom, min(previous.start, candidate.start), max(previous.end, candidate.end), previous.strand)
    return genes


def _stable_seed(*values: str) -> int:
    digest = hashlib.sha256("|".join(values).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _open_pip(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else path.open()


def load_records(
    finemap_dir: Path,
    genes: dict[str, Gene],
    *,
    max_per_panel: int,
    seed: int,
    panels: set[str] | None = None,
) -> tuple[list[Record], dict[str, dict[str, int]]]:
    reservoirs: dict[tuple[str, int], Reservoir] = {}
    raw_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0, "unmatched_gene": 0, "invalid_variant": 0})
    paths = sorted(finemap_dir.glob("*_all.susie_all_pip.tsv.gz"))
    if not paths:
        raise FileNotFoundError(f"No PIP tables found under {finemap_dir}")
    for path in paths:
        panel = path.name.removesuffix("_all.susie_all_pip.tsv.gz")
        if panels is not None and panel not in panels:
            continue
        group = _panel_group(panel)
        positive = Reservoir(max_per_panel, _stable_seed(str(seed), panel, "positive"))
        negative = Reservoir(max_per_panel, _stable_seed(str(seed), panel, "negative"))
        reservoirs[(panel, 1)] = positive
        reservoirs[(panel, 0)] = negative
        with _open_pip(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                try:
                    pip = float(row["susie_pip"])
                    feature = row["feature"].split(".", 1)[0]
                    chrom = row["chr"] if row["chr"].startswith("chr") else f"chr{row['chr']}"
                    pos = int(row["pos"]) - 1
                    ref = row["ref"].upper()
                    alt = row["alt"].upper()
                except (KeyError, TypeError, ValueError):
                    raw_counts[panel]["invalid_variant"] += 1
                    continue
                if feature not in genes:
                    raw_counts[panel]["unmatched_gene"] += 1
                    continue
                gene = genes[feature]
                if chrom != gene.chrom:
                    raw_counts[panel]["invalid_variant"] += 1
                    continue
                if len(ref) != 1 or len(alt) != 1 or ref not in BASES or alt not in BASES or ref == alt:
                    raw_counts[panel]["invalid_variant"] += 1
                    continue
                distance = abs(pos - gene.tss)
                if distance >= WINDOW // 2:
                    continue
                if pip > 0.75:
                    raw_counts[panel]["positive"] += 1
                    positive.add(Record(panel, feature, chrom, pos, ref, alt, pip, 1, group, gene))
                elif pip < 0.01:
                    raw_counts[panel]["negative"] += 1
                    negative.add(Record(panel, feature, chrom, pos, ref, alt, pip, 0, group, gene))
        print(f"loaded {panel}, positive={positive.seen}, negative={negative.seen}, retained={len(positive.items) + len(negative.items)}", flush=True)
    records = []
    retained: dict[str, dict[str, int]] = {}
    for (panel, label), reservoir in reservoirs.items():
        records.extend(reservoir.items)
        retained.setdefault(panel, {})["positive" if label else "negative"] = len(reservoir.items)
    records.sort(key=lambda record: (record.panel, record.label, record.feature, record.pos, record.alt))
    counts = {}
    for panel, values in retained.items():
        counts[panel] = {
            "raw_positive": raw_counts[panel]["positive"],
            "raw_negative": raw_counts[panel]["negative"],
            "unmatched_gene": raw_counts[panel]["unmatched_gene"],
            "invalid_variant": raw_counts[panel]["invalid_variant"],
            "retained_positive": values["positive"],
            "retained_negative": values["negative"],
        }
    return records, counts


def load_sequence_tools(fasta_path: Path):
    from alphagenome.data import genome
    from alphagenome_research.io import fasta as fasta_lib
    from alphagenome_research.model import one_hot_encoder

    return genome, fasta_lib.FastaExtractor(str(fasta_path)), one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)


def _load_head_specs(targets_path: Path):
    from alphagenome_ft.finetune import load_targets_config, prepare_head_specs, register_predefined_heads

    config = json.loads(targets_path.read_text())
    rna = next(head for head in config["heads"] if head["id"].startswith("zemke2023_rna"))
    rna = dict(rna)
    rna["id"] = "zemke2023_rna_human"
    base_config = {"heads": [rna]}
    spec = prepare_head_specs(load_targets_config(base_config))[0]
    register_predefined_heads([spec])

    joint_config = REPO_ROOT / "outputs/v0data/joint-objective-variants/metric-tempered-all-gene-source-specific/datasets.json"
    if joint_config.exists():
        joint = json.loads(joint_config.read_text())
        specs = []
        for dataset in joint.get("datasets", []):
            for source in dataset.get("sources", []):
                source_path = Path(source["targets_config"])
                if source_path.exists():
                    try:
                        specs.extend(prepare_head_specs(load_targets_config(source_path)))
                    except (FileNotFoundError, ValueError):
                        pass
        if specs:
            register_predefined_heads(specs)
    return spec


def _load_model(label: str, checkpoint: Path | None, *, base_checkpoint: Path, targets_path: Path, init_seq_len: int):
    from alphagenome_ft import create_model_with_heads, load_checkpoint

    _load_head_specs(targets_path)
    if checkpoint is None:
        return create_model_with_heads(
            "all_folds",
            heads=["zemke2023_rna_human"],
            checkpoint_path=str(base_checkpoint),
            init_seq_len=init_seq_len,
            include_standard_heads=False,
            pretrained_head_initialization="semantic_neural_accessibility_bootstrap",
            runtime_backbone_param_dtype="bfloat16",
            runtime_backbone_compute_dtype="bfloat16",
        )
    return load_checkpoint(
        checkpoint,
        base_model_version="all_folds",
        base_checkpoint_path=str(base_checkpoint),
        init_seq_len=init_seq_len,
        runtime_backbone_param_dtype="bfloat16",
        runtime_backbone_compute_dtype="bfloat16",
    )


def _encode_window(extractor, encoder, genome, record: Record, *, window: int) -> tuple[np.ndarray, int] | None:
    start = record.gene.tss - window // 2
    end = start + window
    if start < 0:
        return None
    sequence = extractor.extract(genome.Interval(record.chrom, start, end)).upper()
    if len(sequence) != window:
        return None
    offset = record.pos - start
    if offset < 0 or offset >= window or sequence[offset] != record.ref:
        return None
    return encoder.encode(sequence).astype(np.float32), offset


def _predict(model, sequence: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    batch = jnp.asarray(sequence)
    organisms = jnp.zeros((batch.shape[0],), dtype=jnp.int32)
    predictions = model._custom_forward_fn(model._params, model._state, None, batch, organisms)
    values = predictions["zemke2023_rna_human"]["predictions_128bp"]
    return np.asarray(jax.device_get(values), dtype=np.float32)


def score_model(model, records: list[Record], *, extractor, encoder, genome, batch_size: int, window: int) -> list[dict[str, object]]:
    track_names = np.asarray(["ASC", "Endo", "L2_3_IT", "L4_5_IT", "L5_6_NP", "L5_ET", "L5_IT", "L6_CT", "L6_IT_CAR3", "L6b", "LAMP5", "MGC", "ODC", "OPC", "PVALB", "SNCG", "SST", "VIP", "VLMC"])
    indices = _track_indices(track_names)
    groups: dict[tuple[str, int, int, str], list[Record]] = defaultdict(list)
    for record in records:
        groups[(record.chrom, record.gene.tss, record.gene.start, record.feature)].append(record)
    output: list[dict[str, object]] = []
    skipped = 0
    for group_records in groups.values():
        first = group_records[0]
        encoded = _encode_window(extractor, encoder, genome, first, window=window)
        if encoded is None:
            skipped += len(group_records)
            continue
        reference, _ = encoded
        ref_values = _predict(model, reference[None, ...])[0]
        tss_bin = min(ref_values.shape[0] - 1, (window // 2) // 128)
        gene_start_bin = max(0, (first.gene.start - (first.gene.tss - window // 2)) // 128)
        gene_end_bin = min(ref_values.shape[0], (first.gene.end - (first.gene.tss - window // 2) + 127) // 128)
        if gene_end_bin <= gene_start_bin:
            gene_start_bin, gene_end_bin = tss_bin, tss_bin + 1
        for start in range(0, len(group_records), batch_size):
            chunk = group_records[start : start + batch_size]
            alt_sequences = []
            valid = []
            for record in chunk:
                alt = _encode_window(extractor, encoder, genome, record, window=window)
                if alt is None:
                    skipped += 1
                    continue
                alt_sequence, offset = alt
                alt_sequence[offset, :] = 0.0
                alt_sequence[offset, "ACGT".index(record.alt)] = 1.0
                alt_sequences.append(alt_sequence)
                valid.append(record)
            if not valid:
                continue
            alt_values = _predict(model, np.stack(alt_sequences, axis=0))
            for record, values in zip(valid, alt_values):
                track_index = indices[record.group]
                tss_effect = float(np.mean(np.abs(values[tss_bin, track_index] - ref_values[tss_bin, track_index])))
                gene_effect = float(np.mean(np.abs(np.mean(values[gene_start_bin:gene_end_bin, :][:, track_index], axis=0) - np.mean(ref_values[gene_start_bin:gene_end_bin, :][:, track_index], axis=0))))
                output.append({
                    "panel": record.panel,
                    "feature": record.feature,
                    "chrom": record.chrom,
                    "pos": record.pos + 1,
                    "pip": record.pip,
                    "label": record.label,
                    "cell_group": record.group,
                    "distance_bp": record.distance,
                    "distance_bin": record.distance_bin,
                    "tss_effect": tss_effect,
                    "gene_effect": gene_effect,
                })
        if len(output) % 1000 < len(group_records):
            print(f"scored {len(output)} records, skipped={skipped}", flush=True)
    print(f"finished model, scored={len(output)}, skipped={skipped}", flush=True)
    return output


def parse_model(value: str) -> tuple[str, Path | None]:
    label, sep, path = value.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("models must use LABEL=CHECKPOINT, with LABEL=base for the pretrained model")
    return label, None if path == "base" else Path(path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finemap-dir", type=Path, default=DEFAULT_FINEMAP_DIR)
    parser.add_argument("--gtf", type=Path, default=DEFAULT_GTF)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--model", action="append", type=parse_model, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-panel", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--init-seq-len", type=int, default=131072)
    parser.add_argument("--panel", action="append", dest="panels", help="Restrict a smoke run to one or more panel names.")
    args = parser.parse_args()
    if args.window % 128:
        raise ValueError("window must be divisible by 128")
    models = args.model or [(label, path) for label, path in [("base", None), *DEFAULT_CHECKPOINTS.items()]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    genes = load_genes(args.gtf)
    records, counts = load_records(args.finemap_dir, genes, max_per_panel=args.max_per_panel, seed=args.seed, panels=set(args.panels) if args.panels else None)
    (args.output_dir / "sampling_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True))
    (args.output_dir / "benchmark_config.json").write_text(json.dumps({"window": args.window, "positive_pip": ">0.75", "negative_pip": "<0.01", "max_per_panel": args.max_per_panel, "models": {label: str(path) if path else "base+semantic_head" for label, path in models}, "cell_mapping": {"Ast": "ASC", "End": "Endo", "Ext": "mean excitatory Zemke tracks", "IN": "mean inhibitory Zemke tracks", "MG": "MGC", "OD": "ODC", "OPC": "OPC"}}, indent=2, sort_keys=True))
    genome, extractor, encoder = load_sequence_tools(args.fasta)
    all_rows = []
    for label, checkpoint in models:
        print(f"loading model {label}", flush=True)
        model = _load_model(label, checkpoint, base_checkpoint=args.base_checkpoint, targets_path=args.targets, init_seq_len=args.init_seq_len)
        rows = score_model(model, records, extractor=extractor, encoder=encoder, genome=genome, batch_size=args.batch_size, window=args.window)
        for row in rows:
            row["model"] = label
        all_rows.extend(rows)
        del model
    if not all_rows:
        raise RuntimeError("No valid eQTL records were scored")
    columns = list(all_rows[0])
    with (args.output_dir / "scores.tsv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows to {args.output_dir / 'scores.tsv'}")


if __name__ == "__main__":
    main()
