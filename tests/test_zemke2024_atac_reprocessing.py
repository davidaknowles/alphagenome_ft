import gzip

from scripts.zemke2024_atac_reprocessing.aggregate import read_metadata


def test_read_metadata_uses_terminal_barcode_for_deep_and_plain_libraries(tmp_path):
    metadata_path = tmp_path / "metadata.tsv.gz"
    with gzip.open(metadata_path, "wt") as handle:
        handle.write("bacrode\torig.ident\tsubclass\n")
        handle.write("hc1_AAAC-1\thc1\tAstro\n")
        handle.write("hc1_deep_CCCG-1\thc1\tOligo\n")

    groups = read_metadata(metadata_path)

    assert groups == {"hc1": {"AAAC-1": "Astro", "CCCG-1": "Oligo"}}
