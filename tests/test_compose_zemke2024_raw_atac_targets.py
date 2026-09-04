import json

from scripts.zemke2024_atac_reprocessing.compose_raw_atac_targets import (
    compose_raw_atac_targets,
    main,
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


def test_main_writes_composed_manifest(tmp_path, monkeypatch):
    source = {
        "heads": [
            {"id": "atac", "kind": "atac", "targets": [{"label": "old"}]},
            {"id": "rna", "kind": "rna_seq", "targets": [{"label": "rna"}]},
        ]
    }
    raw = {"heads": [{"id": "raw", "kind": "atac", "targets": [{"label": "new"}]}]}
    source_path = tmp_path / "source.json"
    raw_path = tmp_path / "raw.json"
    output_path = tmp_path / "output.json"
    source_path.write_text(json.dumps(source))
    raw_path.write_text(json.dumps(raw))
    monkeypatch.setattr(
        "sys.argv",
        [
            "compose_raw_atac_targets.py",
            "--gene-targets",
            str(source_path),
            "--raw-atac-targets",
            str(raw_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    assert json.loads(output_path.read_text())["heads"][0]["targets"] == [{"label": "new"}]
