from pathlib import Path

import h5py
import numpy as np
import pyBigWig

from alphagenome_ft.finetune.rna_tracks import (
    read_gene_bodies,
    read_pseudobulk_expression,
    write_stranded_gene_body_bigwigs,
)


def test_pseudobulk_gene_body_tracks_preserve_cpm_and_strand(tmp_path: Path):
    h5ad = tmp_path / "tiny.h5ad"
    with h5py.File(h5ad, "w") as handle:
        handle.create_dataset("X", data=np.asarray([[2.0, 1.0]], dtype=np.float32))
        obs = handle.create_group("obs")
        obs.create_dataset("Group", data=np.asarray([b"cell type"]))
        var = handle.create_group("var")
        var.create_dataset("gene_id", data=np.asarray([b"ENSG1", b"ENSG2"]))
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        'chr1\ttest\tgene\t1\t10\t.\t+\t.\tgene_id "ENSG1";\n'
        'chr1\ttest\tgene\t21\t40\t.\t-\t.\tgene_id "ENSG2";\n'
    )

    expression = read_pseudobulk_expression(h5ad)
    genes = read_gene_bodies(gtf, gene_ids=expression.gene_ids, chromosome_sizes={"chr1": 100})
    targets = write_stranded_gene_body_bigwigs(
        expression,
        gene_bodies=genes,
        chromosome_sizes={"chr1": 100},
        output_dir=tmp_path / "tracks",
    )

    assert [target["strand"] for target in targets] == ["+", "-"]
    with pyBigWig.open(targets[0]["path"]) as plus:
        plus_values = np.nan_to_num(plus.values("chr1", 0, 100, numpy=True))
    with pyBigWig.open(targets[1]["path"]) as minus:
        minus_values = np.nan_to_num(minus.values("chr1", 0, 100, numpy=True))
    np.testing.assert_allclose(plus_values.sum(), 2.0 / 3.0 * 1_000_000, rtol=1e-5)
    np.testing.assert_allclose(minus_values.sum(), 1.0 / 3.0 * 1_000_000, rtol=1e-5)
