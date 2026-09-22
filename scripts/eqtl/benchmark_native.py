#!/usr/bin/env python
"""Store native AlphaGenome RNA-seq and CAGE VEP summaries for fine-mapped variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eqtl.benchmark import (  # noqa: E402
    DEFAULT_BASE,
    DEFAULT_FASTA,
    Record,
    load_records_cache,
    load_sequence_tools,
    _encode_window,
)


OUTPUT_NAMES = ("rna_seq", "cage")
SUMMARY_NAMES = ("tss_bin", "gene_span")


def _load_native_model(checkpoint: Path):
    from alphagenome_research.model import dna_model

    return dna_model.create(str(checkpoint))


def _track_table(model) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    from alphagenome.models import dna_output
    from alphagenome_research.model import dna_model

    metadata = model._metadata[dna_model.Organism.HOMO_SAPIENS]
    output_types = {
        "rna_seq": dna_output.OutputType.RNA_SEQ,
        "cage": dna_output.OutputType.CAGE,
    }
    rows: list[dict[str, object]] = []
    indices: dict[str, np.ndarray] = {}
    for output_name in OUTPUT_NAMES:
        output_type = output_types[output_name]
        frame = metadata.get(output_type)
        valid = np.flatnonzero(~np.asarray(metadata.padding[output_type], dtype=bool))
        indices[output_name] = valid.astype(np.int32)
        for channel_index in valid:
            values = frame.iloc[int(channel_index)].to_dict()
            rows.append({
                "track_id": f"{output_name}:{int(channel_index)}",
                "output_type": output_name,
                "channel_index": int(channel_index),
                **{
                    str(key): (None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value))
                    for key, value in values.items()
                },
            })
    return rows, indices


def _make_summary_predictor(model, indices: dict[str, np.ndarray], window: int):
    from alphagenome.models import dna_output
    from alphagenome_research.model import dna_model

    organism = dna_model.Organism.HOMO_SAPIENS
    metadata = model._metadata[organism]
    output_types = {
        "rna_seq": dna_output.OutputType.RNA_SEQ,
        "cage": dna_output.OutputType.CAGE,
    }
    requests = tuple(output_types[name] for name in OUTPUT_NAMES)
    bins = window // 128
    channel_indices = {
        name: jnp.asarray(indices[name], dtype=jnp.int32) for name in OUTPUT_NAMES
    }

    @jax.jit
    def predict_summary(sequences, tss_bin, gene_start_bin, gene_end_bin):
        batch_size = sequences.shape[0]
        predictions = model._predict(
            model._params,
            model._state,
            sequences,
            jnp.zeros((batch_size,), dtype=jnp.int32),
            requested_outputs=requests,
            negative_strand_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
            strand_reindexing=metadata.strand_reindexing,
        )
        summaries = []
        positions = jnp.arange(bins, dtype=jnp.int32)
        gene_mask = (positions >= gene_start_bin) & (positions < gene_end_bin)
        denominator = jnp.maximum(jnp.sum(gene_mask), 1)
        for name in OUTPUT_NAMES:
            profile = predictions[output_types[name]].astype(jnp.float32)
            profile = profile.reshape(batch_size, bins, 128, profile.shape[-1]).mean(axis=2)
            profile = jnp.take(profile, channel_indices[name], axis=-1)
            tss = jnp.take(profile, tss_bin, axis=1)
            gene = jnp.sum(jnp.where(gene_mask[None, :, None], profile, 0.0), axis=1) / denominator
            summaries.append(jnp.stack((tss, gene), axis=1))
        return jnp.concatenate(summaries, axis=-1)

    return predict_summary


def _write_track_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_variant_metadata(path: Path, records: list[Record]) -> None:
    fields = ["record_index", "panel", "feature", "chrom", "pos", "pip", "label", "cell_group", "distance_bp", "distance_bin"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow({
                "record_index": index,
                "panel": record.panel,
                "feature": record.feature,
                "chrom": record.chrom,
                "pos": record.pos + 1,
                "pip": record.pip,
                "label": record.label,
                "cell_group": record.group,
                "distance_bp": record.distance,
                "distance_bin": record.distance_bin,
            })


def _group_records(records: list[Record]):
    grouped: dict[tuple[str, int, int, str], list[tuple[int, Record]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record.chrom, record.gene.tss, record.gene.start, record.feature)].append((index, record))
    return list(grouped.values())


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records_cache(args.records_cache)
    if args.max_records is not None:
        records = records[: args.max_records]
    (args.output_dir / "benchmark_config.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "window_bp": args.window,
        "summary_bin_bp": 128,
        "outputs": list(OUTPUT_NAMES),
        "summaries": list(SUMMARY_NAMES),
        "matrix_shape": [len(records), 2, "n_tracks"],
        "matrix_semantics": "signed alternate minus reference prediction delta; absolute VEP is abs(delta)",
        "max_records": args.max_records,
    }, indent=2) + "\n")
    _write_variant_metadata(args.output_dir / "variants.tsv", records)

    print(f"loading original AlphaGenome checkpoint from {args.checkpoint}", flush=True)
    model = _load_native_model(args.checkpoint)
    track_rows, indices = _track_table(model)
    _write_track_metadata(args.output_dir / "tracks.tsv", track_rows)
    n_tracks = len(track_rows)
    effects_path = args.output_dir / "native_vep_delta.npy"
    completion_path = args.output_dir / "completion.npy"
    expected_shape = (len(records), len(SUMMARY_NAMES), n_tracks)
    if effects_path.exists():
        effects = np.load(effects_path, mmap_mode="r+")
        if effects.shape != expected_shape or effects.dtype != np.float32:
            raise ValueError(f"Existing VEP matrix has incompatible shape/dtype: {effects.shape}/{effects.dtype}")
        completed = np.load(completion_path, mmap_mode="r+")
        if completed.shape != (len(records),):
            raise ValueError(f"Existing completion mask has incompatible shape: {completed.shape}")
        print(f"resuming with {int(completed.sum())} completed records", flush=True)
    else:
        effects = np.lib.format.open_memmap(
            effects_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_shape,
        )
        effects[:] = np.nan
        effects.flush()
        completed = np.lib.format.open_memmap(
            completion_path,
            mode="w+",
            dtype=np.bool_,
            shape=(len(records),),
        )
        completed[:] = False
        completed.flush()

    genome, extractor, encoder = load_sequence_tools(args.fasta)
    groups = _group_records(records)
    print(f"prepared {len(groups)} variant groups", flush=True)
    predict_summary = _make_summary_predictor(model, indices, args.window)
    scored = 0
    skipped = 0
    for group in groups:
        pending = [(index, record) for index, record in group if not completed[index]]
        if not pending:
            continue
        first = group[0][1]
        encoded = _encode_window(extractor, encoder, genome, first, window=args.window)
        if encoded is None:
            for index, _ in pending:
                completed[index] = True
            skipped += len(pending)
            continue
        reference, _ = encoded
        sequence_start = first.gene.tss - args.window // 2
        tss_bin = min(args.window // 128 - 1, (args.window // 2) // 128)
        gene_start_bin = max(0, (first.gene.start - sequence_start) // 128)
        gene_end_bin = min(args.window // 128, (first.gene.end - sequence_start + 127) // 128)
        if gene_end_bin <= gene_start_bin:
            gene_start_bin, gene_end_bin = tss_bin, tss_bin + 1
        reference_summary = np.asarray(jax.device_get(predict_summary(
            jnp.asarray(reference[None, ...]),
            jnp.asarray(tss_bin, dtype=jnp.int32),
            jnp.asarray(gene_start_bin, dtype=jnp.int32),
            jnp.asarray(gene_end_bin, dtype=jnp.int32),
        )[0]), dtype=np.float32)
        valid_alternates = []
        for record_index, record in pending:
            offset = record.pos - sequence_start
            ref_index = "ACGT".index(record.ref)
            if offset < 0 or offset >= args.window or reference[offset, ref_index] < 0.5:
                completed[record_index] = True
                skipped += 1
                continue
            alternate = reference.copy()
            alternate[offset, :] = 0.0
            alternate[offset, "ACGT".index(record.alt)] = 1.0
            valid_alternates.append((record_index, alternate))
        for batch_start in range(0, len(valid_alternates), args.batch_size):
            batch = valid_alternates[batch_start : batch_start + args.batch_size]
            alternate_summaries = np.asarray(jax.device_get(predict_summary(
                jnp.asarray(np.stack([sequence for _, sequence in batch]), dtype=jnp.float32),
                jnp.asarray(tss_bin, dtype=jnp.int32),
                jnp.asarray(gene_start_bin, dtype=jnp.int32),
                jnp.asarray(gene_end_bin, dtype=jnp.int32),
            )), dtype=np.float32)
            for (record_index, _), alternate_summary in zip(batch, alternate_summaries, strict=True):
                effects[record_index] = alternate_summary - reference_summary
                completed[record_index] = True
                scored += 1
                if scored % args.progress_interval == 0:
                    effects.flush()
                    completed.flush()
                    print(f"scored {scored} variants, skipped={skipped}", flush=True)
    effects.flush()
    completed.flush()
    total_scored = int(np.isfinite(effects[:, 0, :]).all(axis=1).sum())
    total_completed = int(completed.sum())
    print(f"finished native AlphaGenome VEPs: scored_total={total_scored}, completed={total_completed}, skipped_this_run={skipped}, tracks={n_tracks}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--window", type=int, default=1_048_576)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=250)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
