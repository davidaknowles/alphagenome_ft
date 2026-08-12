import json
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
from scipy import sparse


def test_aggregate_corrected_pseudobulk_reads_raw_counts(tmp_path: Path):
    source = ad.AnnData(
        X=sparse.csr_matrix(np.full((3, 2), 7.0, dtype=np.float32)),
        obs={"Group": ["b", "a", "b"]},
        var={"gene_id": ["g1", "g2"]},
    )
    source.raw = ad.AnnData(
        X=sparse.csr_matrix([[1, 2], [3, 0], [4, 1]], dtype=np.float32),
        obs=source.obs.copy(),
        var=source.var.copy(),
    )
    input_path = tmp_path / "single_cell.h5ad"
    output_path = tmp_path / "pseudobulk.h5ad"
    source.write_h5ad(input_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/v0data/johansen_rna_reprocessing/aggregate_corrected_pseudobulk.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
    )

    result = ad.read_h5ad(output_path)
    assert result.obs_names.tolist() == ["a", "b"]
    np.testing.assert_allclose(result.X, [[1_000_000, 0], [625_000, 375_000]])
    np.testing.assert_array_equal(result.obs["n_cells"], [1, 2])


def test_rewrite_species_config_uses_corrected_gene_only_supervision(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    targets = {
        "heads": [
            {"id": "atac", "kind": "atac", "targets": []},
            {
                "id": "rna",
                "kind": "rna_seq",
                "gene_supervision": {
                    "path": "old.npz",
                    "loss_weight": 1.0,
                    "coverage_loss_weight": 0.1,
                },
                "targets": [],
            },
        ]
    }
    targets_path = source_dir / "targets.json"
    targets_path.write_text(json.dumps(targets))
    config_path = source_dir / "species.json"
    config_path.write_text(
        json.dumps({"species": [{"name": "human", "targets_config": str(targets_path)}]})
    )
    supervision = tmp_path / "supervision" / "human" / "gene_expression_supervision.npz"
    supervision.parent.mkdir(parents=True)
    np.savez(supervision, cpm=np.ones((1, 1)))
    output = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            "scripts/v0data/johansen_rna_reprocessing/rewrite_species_config.py",
            "--input",
            str(config_path),
            "--supervision-root",
            str(supervision.parents[1]),
            "--output-dir",
            str(output),
            "--correlation-loss-weight",
            "1",
        ],
        check=True,
    )

    rewritten = json.loads((output / "human" / "targets.json").read_text())
    rna = rewritten["heads"][1]
    assert rna["gene_supervision"]["path"] == str(supervision.resolve())
    assert rna["gene_supervision"]["coverage_loss_weight"] == 0
    assert rna["double_centered_correlation_loss_weight"] == 1
