import json
from pathlib import Path

import numpy as np

from scripts.v0data.zemke2023_rna_reprocessing import prepare_published_gene_supervision as module


def test_published_gene_supervision_integrates_tracks_and_restores_all_groups(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.npz"
    np.savez_compressed(
        source,
        gene_ids=np.asarray(["g1", "g2"]),
        chromosomes=np.asarray(["chr1", "chr1"]),
        starts=np.asarray([0, 20]),
        ends=np.asarray([10, 30]),
        strands=np.asarray(["+", "-"]),
        exon_offsets=np.asarray([0, 1, 2]),
        exon_starts=np.asarray([0, 20]),
        exon_ends=np.asarray([10, 30]),
        groups=np.asarray(["A", "B"]),
        cpm=np.zeros((2, 2), dtype=np.float32),
        group_valid=np.asarray([True, False]),
    )
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "heads": [
                    {
                        "id": "rna",
                        "kind": "rna_seq",
                        "targets": [
                            {"label": "A", "path": "a.bw"},
                            {"label": "B", "path": "b.bw"},
                        ],
                    }
                ]
            }
        )
    )

    values = {"a.bw": np.asarray([3.0, 1.0]), "b.bw": np.asarray([2.0, 6.0])}
    monkeypatch.setattr(
        module,
        "_integrate_track",
        lambda path, **kwargs: values[path.name],
    )
    manifest = module.prepare_published_gene_supervision(
        source_supervision=source,
        source_targets=targets,
        output_dir=tmp_path / "output",
        species="test",
    )

    with np.load(manifest["gene_supervision"]) as supervision:
        np.testing.assert_allclose(
            supervision["cpm"],
            [[750_000.0, 250_000.0], [250_000.0, 750_000.0]],
        )
        assert supervision["group_valid"].tolist() == [True, True]
    output_targets = json.loads(Path(manifest["targets"]).read_text())
    rna = output_targets["heads"][0]
    assert rna["gene_supervision"]["coverage_loss_weight"] == 1.0
    assert rna["double_centered_correlation_loss_weight"] == 10.0
