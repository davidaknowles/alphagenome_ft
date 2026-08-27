from pathlib import Path

import numpy as np

from scripts.v0data.zemke2024_rna_reprocessing.prepare_gene_only_target import (
    prepare_gene_only_target,
)


def test_prepare_gene_only_target_uses_direct_cpm_scales_and_masks_missing_groups(
    tmp_path: Path,
) -> None:
    supervision = tmp_path / "genes.npz"
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
        groups=np.asarray(["broad", "missing", "other"]),
        cpm=np.asarray([[10.0, 20.0], [0.0, 0.0], [0.0, 15.0]]),
        group_valid=np.asarray([True, False, True]),
    )
    source = {
        "heads": [
            {"id": "atac", "kind": "atac", "targets": []},
            {
                "id": "zemke2024_all_rna",
                "source": "predefined",
                "kind": "rna_seq",
                "resolutions": [1, 128],
                "targets": [
                    {"label": group, "path": f"{group}.bw", "nonzero_mean": 99.0}
                    for group in ("broad", "missing", "other")
                ],
                "gene_supervision": {
                    "path": "old.npz",
                    "loss_weight": 1.0,
                    "coverage_loss_weight": 1.0,
                },
            },
        ]
    }

    result = prepare_gene_only_target(
        source,
        supervision_path=supervision,
        correlation_loss_weight=1.0,
    )

    rna = result["heads"][1]
    assert rna["resolutions"] == [128]
    assert rna["gene_supervision"]["path"] == str(supervision.resolve())
    assert rna["gene_supervision"]["coverage_loss_weight"] == 0.0
    assert rna["double_centered_correlation_loss_weight"] == 1.0
    np.testing.assert_allclose(
        [target["nonzero_mean"] for target in rna["targets"]],
        [2.0, 1.75, 1.5],
    )
    contract = result["target_contract"]
    assert contract["rna_masked_direct_gene_groups"] == ["missing"]
    assert contract["rna_invalid_scale_fallback"] == 1.75
