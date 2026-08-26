import json
from pathlib import Path

import numpy as np

from scripts.v0data.zemke2023_rna_reprocessing.prepare_gene_only_species import (
    prepare_gene_only_species,
)


def _write_species(root: Path, species: str, cpm: np.ndarray) -> None:
    directory = root / species
    directory.mkdir(parents=True)
    supervision = directory / "gene_expression_supervision.npz"
    np.savez_compressed(
        supervision,
        gene_ids=np.asarray(["a", "b"]),
        chromosomes=np.asarray(["chr1", "chr1"]),
        starts=np.asarray([0, 5]),
        ends=np.asarray([10, 15]),
        strands=np.asarray(["+", "+"]),
        exon_offsets=np.asarray([0, 1, 2]),
        exon_starts=np.asarray([0, 5]),
        exon_ends=np.asarray([10, 15]),
        groups=np.asarray(["both", "first", "masked"]),
        cpm=cpm,
        group_valid=np.asarray([True, True, False]),
    )
    targets = {
        "heads": [
            {"id": "atac", "kind": "atac", "resolutions": [1, 128]},
            {
                "id": "zemke2023_rna",
                "kind": "rna_seq",
                "resolutions": [1, 128],
                "targets": [
                    {"path": f"{group}.bw", "label": group, "strand": "."}
                    for group in ("both", "first", "masked")
                ],
                "gene_supervision": {
                    "path": str(supervision),
                    "loss_weight": 1.0,
                    "coverage_loss_weight": 1.0,
                },
            },
        ]
    }
    (directory / "targets.json").write_text(json.dumps(targets))


def test_prepare_gene_only_species_pools_synthetic_scales(tmp_path: Path) -> None:
    supervision_root = tmp_path / "supervision"
    _write_species(
        supervision_root,
        "first",
        np.asarray([[10.0, 20.0], [10.0, 0.0], [0.0, 0.0]]),
    )
    _write_species(
        supervision_root,
        "second",
        np.asarray([[30.0, 0.0], [20.0, 0.0], [0.0, 0.0]]),
    )
    output_dir = tmp_path / "output"

    result = prepare_gene_only_species(
        {"species": [{"name": "first"}, {"name": "second"}]},
        supervision_root=supervision_root,
        output_dir=output_dir,
    )

    np.testing.assert_allclose(result["rna_pooled_nonzero_means"], [2.5, 1.5, 2.0])
    for entry in result["species"]:
        assert entry["direct_gene_valid_groups"] == 2
        config = json.loads(Path(entry["targets_config"]).read_text())
        rna = next(head for head in config["heads"] if head["id"] == "zemke2023_rna")
        assert rna["resolutions"] == [128]
        assert rna["gene_supervision"]["coverage_loss_weight"] == 0.0
        assert rna["double_centered_correlation_loss_weight"] == 1.0
        np.testing.assert_allclose(
            [target["nonzero_mean"] for target in rna["targets"]],
            [2.5, 1.5, 2.0],
        )
