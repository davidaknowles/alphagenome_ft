import gzip
import json
from pathlib import Path

import numpy as np
from scipy import io, sparse

from scripts.v0data.zemke2023_rna_reprocessing.prepare_gene_supervision import (
    prepare_gene_supervision,
    read_barcode_groups,
)


def _write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as handle:
        handle.write(text)


def test_read_barcode_groups_maps_release_punctuation_and_excludes_chc(tmp_path: Path):
    metadata = tmp_path / "metadata.tsv.gz"
    _write_gzip(
        metadata,
        "cell\tsubclass\ncell1\tL2/3 IT\ncell2\tChC\ncell3\tASC\n",
    )

    groups, excluded = read_barcode_groups(
        metadata,
        target_groups=("ASC", "L2_3_IT"),
    )

    assert groups == {"cell1": "L2_3_IT", "cell3": "ASC"}
    assert excluded == {"ChC": 1}


def test_prepare_gene_supervision_preserves_tracks_and_adds_raw_cpm(tmp_path: Path):
    matrix = sparse.coo_matrix(
        np.asarray(
            [
                [10, 0, 4, 3],
                [0, 6, 2, 1],
                [5, 4, 0, 7],
            ],
            dtype=np.int64,
        )
    )
    matrix_path = tmp_path / "matrix.mtx"
    io.mmwrite(matrix_path, matrix)
    barcodes = tmp_path / "barcodes.tsv.gz"
    features = tmp_path / "features.tsv.gz"
    metadata = tmp_path / "metadata.tsv.gz"
    _write_gzip(barcodes, "cell1\ncell2\ncell3\ncell4\n")
    _write_gzip(features, "Gene.1\nGene.2\nGene3\n")
    _write_gzip(
        metadata,
        "cell\tsubclass\ncell1\tA\ncell2\tB\ncell3\tA\ncell4\tChC\n",
    )
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "target_contract": {"rna": "published_native_signal"},
                "heads": [
                    {
                        "id": "atac",
                        "kind": "atac",
                        "targets": [{"label": "A", "path": "a.bw"}],
                    },
                    {
                        "id": "rna",
                        "kind": "rna_seq",
                        "resolutions": [1, 128],
                        "targets": [
                            {"label": "A", "path": "a-rna.bw"},
                            {"label": "B", "path": "b-rna.bw"},
                        ],
                    },
                ],
            }
        )
    )
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "A" * 1000 + "\n")
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        'chr1\ttest\texon\t11\t20\t.\t+\t.\tgene_id "Gene.1"; gene_name "Gene.1";\n'
        'chr1\ttest\texon\t31\t50\t.\t-\t.\tgene_id "Gene.2"; gene_name "Gene.2";\n'
    )

    output_dir = tmp_path / "output"
    manifest = prepare_gene_supervision(
        matrix_path=matrix_path,
        barcode_path=barcodes,
        feature_path=features,
        metadata_path=metadata,
        targets_path=targets,
        gtf_path=gtf,
        fasta_path=fasta,
        output_dir=output_dir,
        species="test",
        minimum_gene_coverage=0.5,
        correlation_loss_weight=10.0,
    )

    output_targets = json.loads((output_dir / "targets.json").read_text())
    assert output_targets["heads"][0] == json.loads(targets.read_text())["heads"][0]
    rna = output_targets["heads"][1]
    assert rna["resolutions"] == [1, 128]
    assert rna["targets"] == json.loads(targets.read_text())["heads"][1]["targets"]
    assert rna["gene_supervision"]["coverage_loss_weight"] == 1.0
    assert rna["double_centered_correlation_loss_weight"] == 10.0
    with np.load(output_dir / "gene_expression_supervision.npz") as supervision:
        assert supervision["groups"].tolist() == ["A", "B"]
        assert supervision["gene_ids"].tolist() == ["Gene.1", "Gene.2"]
        assert np.allclose(supervision["cpm"][0], [14 / 21 * 1e6, 2 / 21 * 1e6])
        assert np.allclose(supervision["cpm"][1], [0, 6 / 10 * 1e6])
    assert manifest["excluded_groups"] == {"ChC": 1}
    assert manifest["matched_genes"] == 2


def test_prepare_gene_supervision_masks_explicitly_unsupported_group(tmp_path: Path):
    matrix = sparse.coo_matrix(np.asarray([[4, 6]], dtype=np.int64))
    matrix_path = tmp_path / "matrix.mtx"
    io.mmwrite(matrix_path, matrix)
    barcodes = tmp_path / "barcodes.tsv.gz"
    features = tmp_path / "features.tsv.gz"
    metadata = tmp_path / "metadata.tsv.gz"
    _write_gzip(barcodes, "cell1\ncell2\n")
    _write_gzip(features, "Gene.1\n")
    _write_gzip(metadata, "cell\tsubclass\ncell1\tA\ncell2\tA\n")
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
                            {"label": "unreleased", "path": "unreleased.bw"},
                        ],
                    }
                ]
            }
        )
    )
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "A" * 1000 + "\n")
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        'chr1\ttest\texon\t11\t20\t.\t+\t.\tgene_id "Gene.1"; gene_name "Gene.1";\n'
    )

    manifest = prepare_gene_supervision(
        matrix_path=matrix_path,
        barcode_path=barcodes,
        feature_path=features,
        metadata_path=metadata,
        targets_path=targets,
        gtf_path=gtf,
        fasta_path=fasta,
        output_dir=tmp_path / "output",
        species="test",
        unsupported_groups=("unreleased",),
    )

    with np.load(manifest["gene_supervision"]) as supervision:
        assert supervision["groups"].tolist() == ["A", "unreleased"]
        assert supervision["group_valid"].tolist() == [True, False]
        assert supervision["cpm"][:, 0].tolist() == [1_000_000.0, 0.0]
    assert manifest["direct_gene_groups"] == ["A"]
    assert manifest["unsupported_direct_gene_groups"] == ["unreleased"]
