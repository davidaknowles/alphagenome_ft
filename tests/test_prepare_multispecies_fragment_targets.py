from pathlib import Path
import runpy

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/allen_atac_reprocessing/prepare_multispecies_targets.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_replace_atac_head_filters_aligned_atac_and_rna_channels():
    source = {
        "heads": [
            {"id": "atac", "targets": [{"label": "a"}, {"label": "b"}]},
            {
                "id": "rna",
                "targets": [
                    {"label": "a (+)"},
                    {"label": "a (-)"},
                    {"label": "b (+)"},
                    {"label": "b (-)"},
                ],
            },
        ]
    }
    replacement = {
        "heads": [
            {"id": "replacement", "targets": [{"label": "a"}, {"label": "b"}]}
        ]
    }

    result = MODULE["replace_atac_head"](
        source, replacement, species="human", retained_groups=["b"]
    )

    assert [target["label"] for target in result["heads"][0]["targets"]] == ["b"]
    assert [target["label"] for target in result["heads"][1]["targets"]] == [
        "b (+)",
        "b (-)",
    ]


def test_filter_gene_supervision_subsets_groups_and_cpm(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "output.npz"
    np.savez_compressed(
        source,
        groups=np.asarray(["a", "b", "c"]),
        cpm=np.arange(12, dtype=np.float32).reshape(3, 4),
        gene_ids=np.asarray(["g1", "g2", "g3", "g4"]),
    )

    MODULE["filter_gene_supervision"](source, output, ["c", "a"])

    with np.load(output, allow_pickle=False) as result:
        assert result["groups"].tolist() == ["c", "a"]
        np.testing.assert_array_equal(
            result["cpm"], np.asarray([[8, 9, 10, 11], [0, 1, 2, 3]])
        )
        assert result["gene_ids"].tolist() == ["g1", "g2", "g3", "g4"]


def test_read_fragment_depths_uses_stored_whole_genome_totals(tmp_path):
    np.savez_compressed(
        tmp_path / "chr1.npz",
        groups=np.asarray(["a", "b"]),
        total_fragments=np.asarray([12, 34]),
    )

    assert MODULE["read_fragment_depths"](tmp_path) == {"a": 12, "b": 34}
