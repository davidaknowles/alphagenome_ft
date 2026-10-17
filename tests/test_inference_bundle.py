import json
from pathlib import Path

import numpy as np
import pytest
from alphagenome.data import genome

from alphagenome_ft.finetune.config import head_layout_signature, load_targets_config, prepare_head_specs
from alphagenome_ft.inference_bundle import (
    bundle_head_specs,
    export_inference_bundle,
    gene_expression,
    head_organism_index,
    load_gene_annotation,
    read_bundle_heads,
)


def _write_gene_supervision(path: Path) -> None:
    np.savez(
        path,
        gene_ids=np.asarray(["G1", "G2"]),
        chromosomes=np.asarray(["chr1", "chr1"]),
        starts=np.asarray([100, 900]),
        ends=np.asarray([400, 2000]),
        strands=np.asarray(["+", "-"]),
        exon_offsets=np.asarray([0, 2, 3]),
        exon_starts=np.asarray([100, 300, 900]),
        exon_ends=np.asarray([164, 400, 1000]),
        groups=np.asarray(["A", "B"]),
        cpm=np.ones((2, 2), dtype=np.float32),
        group_valid=np.asarray([True, True]),
    )


def _write_training_run(tmp_path: Path) -> tuple[Path, Path]:
    for name in ("a_atac.bw", "b_atac.bw", "a_rna.bw", "b_rna.bw"):
        (tmp_path / name).write_bytes(b"")
    _write_gene_supervision(tmp_path / "genes.npz")
    heads = []
    for species in ("human", "mouse"):
        heads_for_species = [
            {
                "id": f"toy_atac_{species}",
                "source": "predefined",
                "kind": "atac",
                "resolutions": [1, 128],
                "apply_squashing": False,
                "targets": [
                    {"path": str(tmp_path / "a_atac.bw"), "label": "A", "strand": "."},
                    {"path": str(tmp_path / "b_atac.bw"), "label": "B", "strand": "."},
                ],
            },
            {
                "id": f"toy_rna_{species}",
                "source": "predefined",
                "kind": "rna_seq",
                "resolutions": [128],
                "apply_squashing": True,
                "gene_supervision": {"path": str(tmp_path / "genes.npz"), "coverage_loss_weight": 0.0},
                "targets": [
                    {"path": str(tmp_path / "a_rna.bw"), "label": "A", "strand": ".", "nonzero_mean": 0.5},
                    {"path": str(tmp_path / "b_rna.bw"), "label": "B", "strand": ".", "nonzero_mean": 2.0},
                ],
            },
        ]
        (tmp_path / f"{species}.json").write_text(json.dumps({"heads": heads_for_species}))
        heads.extend(heads_for_species)
    datasets = {
        "datasets": [
            {
                "name": "toy",
                "sources": [
                    {"name": "human", "fasta": "/ref/hg38.fa", "targets_config": str(tmp_path / "human.json")},
                    {
                        "name": "mouse",
                        "organism": "MUS_MUSCULUS",
                        "fasta": "/ref/mm10.fa",
                        "targets_config": str(tmp_path / "mouse.json"),
                    },
                ],
            }
        ]
    }
    (tmp_path / "datasets.json").write_text(json.dumps(datasets))
    checkpoint = tmp_path / "run" / "best"
    (checkpoint / "checkpoint").mkdir(parents=True)
    (checkpoint / "checkpoint" / "params").write_bytes(b"weights")
    (checkpoint / "config.json").write_text(json.dumps({"custom_heads": [head["id"] for head in heads]}))
    return checkpoint, tmp_path / "datasets.json"


def test_bundle_reproduces_head_layout_without_targets(tmp_path: Path):
    checkpoint, datasets = _write_training_run(tmp_path)
    bundle = export_inference_bundle(checkpoint, datasets, tmp_path / "bundle", provenance={"run": "toy"})

    heads = read_bundle_heads(bundle)
    assert [head["id"] for head in heads] == json.loads((checkpoint / "config.json").read_text())["custom_heads"]
    assert all("path" not in target for head in heads for target in head["targets"])
    assert (bundle / "genes" / "toy_rna_human.npz").exists()
    assert "checkpoint/params" in (bundle / "SHA256SUMS").read_text()

    for name in ("a_atac.bw", "b_atac.bw", "a_rna.bw", "b_rna.bw"):
        (tmp_path / name).unlink()
    original = []
    for species in ("human", "mouse"):
        config = json.loads((tmp_path / f"{species}.json").read_text())
        original.extend(prepare_head_specs(config, require_target_files=False))
    restored = bundle_head_specs(bundle)
    assert head_layout_signature(restored) == head_layout_signature(original)
    for before, after in zip(original, restored):
        assert after.config == before.config
        for organism, metadata in before.metadata.items():
            field = before.config.output_type.name.lower()
            frame = getattr(metadata, field)
            restored_frame = getattr(after.metadata[organism], field)
            assert restored_frame.equals(frame)

    organisms = {head["id"]: head_organism_index(head) for head in heads}
    assert organisms["toy_rna_human"] != organisms["toy_rna_mouse"]


def test_bundle_refuses_existing_directory(tmp_path: Path):
    checkpoint, datasets = _write_training_run(tmp_path)
    (tmp_path / "bundle").mkdir()
    with pytest.raises(FileExistsError):
        export_inference_bundle(checkpoint, datasets, tmp_path / "bundle")


def test_training_target_paths_are_still_required_by_default(tmp_path: Path):
    _write_training_run(tmp_path)
    (tmp_path / "a_rna.bw").unlink()
    with pytest.raises(FileNotFoundError):
        prepare_head_specs(load_targets_config(tmp_path / "human.json"))


def test_gene_expression_sums_exon_weighted_bins(tmp_path: Path):
    checkpoint, datasets = _write_training_run(tmp_path)
    bundle = export_inference_bundle(checkpoint, datasets, tmp_path / "bundle")
    annotation = load_gene_annotation(bundle, "toy_rna_human")
    window = genome.Interval("chr1", 0, 1024)
    prediction = np.zeros((8, 2), dtype=np.float32)
    prediction[0] = [1.0, 10.0]
    prediction[2] = [2.0, 20.0]
    prediction[3] = [3.0, 30.0]

    gene_ids, values = gene_expression(prediction, annotation, window)

    # Only G1 lies inside the window. Its exons cover 28 bp of bin 0, 36 bp of bin 1,
    # 84 bp of bin 2, and 16 bp of bin 3.
    assert gene_ids == ["G1"]
    expected = (28 / 128) * prediction[0] + (84 / 128) * prediction[2] + (16 / 128) * prediction[3]
    np.testing.assert_allclose(values[0], expected, rtol=1e-6)
