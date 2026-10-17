"""Portable inference bundles for fine-tuned AlphaGenome checkpoints.

A bundle contains the trained heads and backbone adapters, the output track
layout of every head, and the exon geometry used for gene-level RNA
supervision. It does not contain the pretrained AlphaGenome weights or the
training BigWigs, so it can be loaded wherever the base model is available.

Layout::

    config.json        checkpoint configuration written by save_checkpoint
    checkpoint/        Orbax parameters (heads and adapters only)
    heads.json         head definitions without target paths, plus provenance
    genes/<head>.npz   exon geometry and pseudobulk CPM for gene-supervised heads
    SHA256SUMS         checksums of every file above
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BUNDLE_FORMAT_VERSION = 1
HEADS_FILE = "heads.json"
CHECKSUM_FILE = "SHA256SUMS"
DEFAULT_ORGANISM = "HOMO_SAPIENS"


def _portable_head(head: Mapping[str, Any], gene_path: str | None) -> dict[str, Any]:
    """Drop target paths from a targets-config head and point gene supervision at the bundle."""
    if "target_transform" in head:
        raise NotImplementedError(
            f'Head "{head["id"]}" uses a target transform, which bundles do not yet carry.'
        )
    entry = copy.deepcopy(dict(head))
    entry["targets"] = [
        {key: value for key, value in target.items() if key != "path"}
        for target in head["targets"]
    ]
    if gene_path is None:
        entry.pop("gene_supervision", None)
    else:
        entry["gene_supervision"] = {**head["gene_supervision"], "path": gene_path}
    return entry


def write_checksums(bundle_dir: Path) -> None:
    lines = []
    for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        if path.name == CHECKSUM_FILE:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(bundle_dir).as_posix()}")
    (bundle_dir / CHECKSUM_FILE).write_text("\n".join(lines) + "\n")


def export_inference_bundle(
    checkpoint_dir: str | Path,
    dataset_config: str | Path,
    output_dir: str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write an inference bundle for a checkpoint trained from a multi-dataset config.

    Args:
        checkpoint_dir: Saved checkpoint directory containing ``config.json`` and ``checkpoint/``.
        dataset_config: The ``datasets.json`` used for training; its target configs define
            the head track layouts and must still be readable.
        output_dir: New directory for the bundle. It must not already exist.
        provenance: Optional metadata recorded in ``heads.json``.

    Returns:
        The bundle directory.
    """
    from alphagenome_ft.finetune.config import load_targets_config, prepare_head_specs

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    config = json.loads((checkpoint_dir / "config.json").read_text())
    datasets = json.loads(Path(dataset_config).read_text())["datasets"]
    if output_dir.exists():
        raise FileExistsError(f"Bundle directory already exists: {output_dir}")

    heads: dict[str, dict[str, Any]] = {}
    gene_sources: dict[str, Path] = {}
    for dataset in datasets:
        for source in dataset["sources"]:
            targets = load_targets_config(Path(source["targets_config"]))
            prepare_head_specs(targets)
            route = {
                "dataset": dataset["name"],
                "native_source": source["name"],
                "organism": str(source.get("organism", DEFAULT_ORGANISM)),
                "reference": Path(source["fasta"]).name if "fasta" in source else None,
            }
            for head in targets["heads"]:
                head_id = head["id"]
                gene_path = f"genes/{head_id}.npz" if "gene_supervision" in head else None
                entry = _portable_head(head, gene_path)
                if head_id in heads:
                    previous = {k: v for k, v in heads[head_id].items() if k != "sources"}
                    if previous != entry:
                        raise ValueError(f'Head "{head_id}" differs between native sources.')
                    heads[head_id]["sources"].append(route)
                    continue
                entry["sources"] = [route]
                heads[head_id] = entry
                if gene_path is not None:
                    gene_sources[head_id] = Path(head["gene_supervision"]["path"])

    custom_heads = list(config["custom_heads"])
    if set(heads) != set(custom_heads):
        raise ValueError(
            "Dataset config heads do not match the checkpoint: "
            f"missing {sorted(set(custom_heads) - set(heads))}, "
            f"extra {sorted(set(heads) - set(custom_heads))}."
        )

    output_dir.mkdir(parents=True)
    shutil.copytree(checkpoint_dir / "checkpoint", output_dir / "checkpoint")
    shutil.copy2(checkpoint_dir / "config.json", output_dir / "config.json")
    if gene_sources:
        (output_dir / "genes").mkdir()
    for head_id, path in gene_sources.items():
        shutil.copy2(path, output_dir / "genes" / f"{head_id}.npz")
    payload = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "provenance": dict(provenance or {}),
        "heads": [heads[head_id] for head_id in custom_heads],
    }
    (output_dir / HEADS_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    write_checksums(output_dir)
    return output_dir


def read_bundle_heads(bundle_dir: str | Path) -> list[dict[str, Any]]:
    """Return the head definitions of a bundle, with gene paths resolved."""
    bundle_dir = Path(bundle_dir).resolve()
    payload = json.loads((bundle_dir / HEADS_FILE).read_text())
    if payload.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError(f"Unsupported bundle format {payload.get('format_version')!r}.")
    heads = payload["heads"]
    for head in heads:
        if "gene_supervision" in head:
            head["gene_supervision"]["path"] = str(bundle_dir / head["gene_supervision"]["path"])
    return heads


def bundle_head_specs(bundle_dir: str | Path, head_ids: Sequence[str] | None = None):
    """Build HeadSpec objects for a bundle without any training target files."""
    from alphagenome_ft.finetune.config import prepare_head_specs

    heads = read_bundle_heads(bundle_dir)
    if head_ids is not None:
        wanted = set(head_ids)
        heads = [head for head in heads if head["id"] in wanted]
        missing = wanted - {head["id"] for head in heads}
        if missing:
            raise KeyError(f"Heads not in bundle: {sorted(missing)}")
    return prepare_head_specs({"heads": heads}, require_target_files=False)


def load_inference_bundle(
    bundle_dir: str | Path,
    *,
    base_checkpoint_path: str | Path | None = None,
    **load_kwargs: Any,
):
    """Register the bundle's heads and load its checkpoint onto the base model.

    Args:
        bundle_dir: Bundle directory written by :func:`export_inference_bundle`.
        base_checkpoint_path: Local pretrained AlphaGenome ``all_folds`` checkpoint. If
            omitted, the base model is downloaded through Kaggle.
        **load_kwargs: Passed to :func:`alphagenome_ft.load_checkpoint`, for example
            ``runtime_backbone_param_dtype="bfloat16"``.
    """
    from alphagenome_ft.custom_model import load_checkpoint
    from alphagenome_ft.finetune.train import register_predefined_heads

    register_predefined_heads(bundle_head_specs(bundle_dir))
    return load_checkpoint(
        Path(bundle_dir).resolve(),
        base_checkpoint_path=None if base_checkpoint_path is None else str(base_checkpoint_path),
        **load_kwargs,
    )


def head_organism_index(head: Mapping[str, Any]) -> int:
    """Return the AlphaGenome organism index a head was trained with."""
    from alphagenome.models import dna_client
    from alphagenome_research.model import dna_model

    organisms = {route["organism"] for route in head["sources"]}
    if len(organisms) != 1:
        raise ValueError(f'Head "{head["id"]}" was trained with several organisms: {organisms}.')
    return int(dna_model.convert_to_organism_index(dna_client.Organism[organisms.pop()]))


def predict_heads(model, sequences, organism_index: int) -> dict[str, Any]:
    """Run every custom head on one-hot sequences of shape (batch, length, 4).

    Returns a mapping from head name to its prediction dictionary, for example
    ``{"predictions_128bp": array(batch, length // 128, tracks)}``.
    """
    import jax
    import jax.numpy as jnp

    sequences = jnp.asarray(sequences)
    organisms = jnp.full((sequences.shape[0],), organism_index, dtype=jnp.int32)
    result = model._custom_forward_fn(model._params, model._state, None, sequences, organisms)
    predictions = result[0] if isinstance(result, tuple) else result
    return jax.device_get({name: predictions[name] for name in model._custom_heads})


def load_gene_annotation(bundle_dir: str | Path, head_id: str):
    """Load the exon geometry a gene-supervised RNA head was trained against."""
    from alphagenome_ft.finetune.data import GeneExpressionSupervision

    (spec,) = bundle_head_specs(bundle_dir, [head_id])
    if spec.gene_supervision_path is None:
        raise ValueError(f'Head "{head_id}" has no gene supervision.')
    return GeneExpressionSupervision(spec.gene_supervision_path, spec)


def gene_expression(prediction_128bp, annotation, window, *, head_id: str = "head"):
    """Sum 128 bp RNA predictions over exons, as in the gene-level training loss.

    Args:
        prediction_128bp: Predictions for one window, shape (length // 128, tracks).
        annotation: Result of :func:`load_gene_annotation` for the same head.
        window: ``alphagenome.data.genome.Interval`` whose start is the first input base.
        head_id: Name used for intermediate keys only.

    Returns:
        ``(gene_ids, values)`` for genes whose full span lies in the window, where
        ``values`` has shape (genes, groups) and is on the training CPM scale.
    """
    import jax.numpy as jnp

    from alphagenome_ft.finetune.train import _gene_expression_prediction

    indices = annotation.indices_for_window(window)
    prediction_128bp = np.asarray(prediction_128bp)
    arrays = annotation.arrays_for_window(
        window,
        sequence_length=prediction_128bp.shape[0] * 128,
        max_genes=max(len(indices), 1),
    )
    batch = {
        f"gene_{key}_{head_id}": jnp.asarray(arrays[key])[None]
        for key in ("weights", "targets", "strands")
    }
    values = _gene_expression_prediction(
        {"predictions_128bp": jnp.asarray(prediction_128bp)[None]}, batch, head_id
    )
    gene_ids = [annotation.gene_ids[index] for index in indices]
    return gene_ids, np.asarray(values[0, : len(indices)])
