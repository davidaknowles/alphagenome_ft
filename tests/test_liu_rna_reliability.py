from pathlib import Path

import numpy as np

from scripts.v0data.liu_hdma.audit_rna_reliability import chromosome_reliability


def test_chromosome_reliability_uses_the_modeled_gene_map(tmp_path: Path) -> None:
    supervision = tmp_path / "genes.npz"
    np.savez_compressed(
        supervision,
        gene_ids=np.asarray(["gene_a", "gene_b", "gene_c", "gene_d"]),
        chromosomes=np.asarray(["chr8", "chr8", "chr9", "chr9"]),
    )
    first = np.asarray(
        [[1.0, 4.0, 2.0, 8.0], [5.0, 2.0, 7.0, 1.0], [3.0, 6.0, 1.0, 5.0]]
    )
    second = np.asarray(
        [[1.2, 3.8, 2.1, 7.9], [4.8, 2.2, 6.8, 1.2], [3.1, 5.9, 1.1, 4.9]]
    )

    result = chromosome_reliability(
        first,
        second,
        ("gene_a.1", "gene_b.2", "gene_c", "unmodeled"),
        supervision,
    )

    assert set(result) == {"chr8"}
    assert result["chr8"]["genes"] == 2
    assert result["chr8"]["raw_cpm_double_centered_r"] > 0.99
