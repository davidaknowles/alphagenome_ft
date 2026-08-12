from pathlib import Path

import numpy as np

from scripts.v0data.liu_hdma.audit_split_target_rank import audit_chromosomes


def test_split_target_rank_restricts_each_chromosome(tmp_path: Path) -> None:
    path = tmp_path / "supervision.npz"
    genes = np.asarray(["chr8", "chr8", "chr8", "chr9", "chr9", "chr9"])
    groups = np.asarray([-1.0, 0.0, 1.0])
    latent = np.asarray([-2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    cpm = (latent[:, None] @ groups[None, :] + 10.0).T
    np.savez_compressed(path, chromosomes=genes, cpm=cpm)

    result = audit_chromosomes(path, ("chr8", "chr9"))

    assert result["chromosomes"]["chr8"]["raw_cpm"]["observations"] == 3
    assert result["chromosomes"]["chr9"]["raw_cpm"]["observations"] == 3
    assert result["chromosomes"]["chr8"]["raw_cpm"]["rank_for_correlation"]["0.8"] == 1
