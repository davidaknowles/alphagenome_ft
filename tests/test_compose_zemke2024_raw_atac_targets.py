from scripts.zemke2024_atac_reprocessing.compose_raw_atac_targets import (
    compose_raw_atac_targets,
)


def test_compose_raw_atac_preserves_direct_rna_head_and_replaces_atac():
    source = {
        "target_contract": {"rna_gene_only": "direct CPM"},
        "heads": [
            {
                "id": "zemke2024_all_atac",
                "kind": "atac",
                "targets": [{"label": "OldA"}, {"label": "OldB"}],
            },
            {
                "id": "zemke2024_all_rna",
                "kind": "rna_seq",
                "targets": [{"label": "Astro_all", "gene_supervision": {}}],
            },
        ],
    }
    raw = {
        "heads": [
            {
                "id": "raw_atac",
                "kind": "atac",
                "resolutions": [1, 128],
                "targets": [{"label": "Astro", "path": "astro.bw", "nonzero_mean": 0.1}],
            }
        ]
    }

    result = compose_raw_atac_targets(source, raw)

    assert result["heads"][0]["id"] == "zemke2024_all_atac"
    assert result["heads"][0]["targets"] == raw["heads"][0]["targets"]
    assert result["heads"][1] == source["heads"][1]
    assert result["target_contract"]["atac_raw_channels"] == 1
    assert result["target_contract"]["atac_replaced_published_channels"] == 2
