from pathlib import Path

import h5py
import numpy as np
import pyBigWig
from alphagenome.data import genome

from alphagenome_ft.finetune.config import HeadSpec, TrackInfo
from alphagenome_ft.finetune.data import GeneExpressionSupervision
from alphagenome_ft.finetune.rna_tracks import (
    read_gene_exons,
    read_pseudobulk_expression,
    write_gene_expression_supervision,
    write_stranded_exon_bigwigs,
)


def _tiny_expression(tmp_path: Path):
    h5ad = tmp_path / "tiny.h5ad"
    with h5py.File(h5ad, "w") as handle:
        handle.create_dataset("X", data=np.asarray([[2.0, 1.0]], dtype=np.float32))
        obs = handle.create_group("obs")
        obs.create_dataset("Group", data=np.asarray([b"cell type"]))
        var = handle.create_group("var")
        var.create_dataset("gene_id", data=np.asarray([b"ENSG1", b"ENSG2"]))
    return read_pseudobulk_expression(h5ad)


def _tiny_genes(tmp_path: Path, expression):
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        'chr1\ttest\texon\t1\t10\t.\t+\t.\tgene_id "ENSG1";\n'
        'chr1\ttest\texon\t6\t15\t.\t+\t.\tgene_id "ENSG1";\n'
        'chr1\ttest\texon\t31\t40\t.\t+\t.\tgene_id "ENSG1";\n'
        'chr1\ttest\texon\t61\t80\t.\t-\t.\tgene_id "ENSG2";\n'
    )
    return read_gene_exons(
        gtf,
        gene_ids=expression.gene_ids,
        chromosome_sizes={"chr1": 256},
    )


def test_pseudobulk_exon_tracks_preserve_cpm_strand_and_introns(tmp_path: Path):
    expression = _tiny_expression(tmp_path)
    genes = _tiny_genes(tmp_path, expression)
    assert genes["ENSG1"].exons == ((0, 15), (30, 40))

    targets = write_stranded_exon_bigwigs(
        expression,
        genes=genes,
        chromosome_sizes={"chr1": 256},
        output_dir=tmp_path / "tracks",
    )

    assert [target["strand"] for target in targets] == ["+", "-"]
    assert all(float(target["nonzero_mean"]) > 0 for target in targets)
    with pyBigWig.open(targets[0]["path"]) as plus:
        plus_values = np.nan_to_num(plus.values("chr1", 0, 256, numpy=True))
    with pyBigWig.open(targets[1]["path"]) as minus:
        minus_values = np.nan_to_num(minus.values("chr1", 0, 256, numpy=True))
    np.testing.assert_allclose(plus_values.sum(), 2.0 / 3.0 * 1_000_000, rtol=1e-5)
    np.testing.assert_allclose(minus_values.sum(), 1.0 / 3.0 * 1_000_000, rtol=1e-5)
    assert not plus_values[15:30].any()


def test_gene_supervision_builds_128bp_exon_weights_and_targets(tmp_path: Path):
    expression = _tiny_expression(tmp_path)
    genes = _tiny_genes(tmp_path, expression)
    artifact = tmp_path / "genes.npz"
    write_gene_expression_supervision(artifact, expression, genes=genes)
    tracks = (
        TrackInfo("cell type (+)", tmp_path / "plus.bw", "+"),
        TrackInfo("cell type (-)", tmp_path / "minus.bw", "-"),
    )
    spec = HeadSpec("rna", "predefined", "rna_seq", tracks, gene_supervision_path=artifact)
    supervision = GeneExpressionSupervision(artifact, spec)
    window = genome.Interval("chr1", 0, 256)
    arrays = supervision.arrays_for_window(window, sequence_length=256, max_genes=3)

    assert arrays["valid"].tolist() == [True, True, False]
    assert arrays["strands"].tolist() == [0, 1, 0]
    np.testing.assert_allclose(arrays["weights"][:, 0].sum(), 25 / 128)
    np.testing.assert_allclose(arrays["weights"][:, 1].sum(), 20 / 128)
    np.testing.assert_allclose(arrays["targets"][0], [2.0 / 3.0 * 1_000_000])
    np.testing.assert_allclose(arrays["targets"][1], [1.0 / 3.0 * 1_000_000])
