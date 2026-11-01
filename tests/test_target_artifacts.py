import pandas as pd

from alphagenome_ft.finetune.target_artifacts import blacklist_regions, extreme_blocks


def test_blacklist_regions_merge_padded_flagged_bins():
    top = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr2"],
        "start": [1000, 1128, 50000, 0],
        "end": [1128, 1256, 50128, 128],
        "minimum": [5000.0, 3000.0, 10.0, 4000.0],
        "mean": [1.0, 1.0, 1.0, 1.0],
    })
    regions = blacklist_regions(top, threshold=1000.0, pad=100)
    assert regions.values.tolist() == [["chr1", 900, 1356], ["chr2", 0, 228]]


def test_extreme_blocks_use_median_multiple():
    blocks = pd.DataFrame({"chrom": "chr1", "start": [0, 10, 20, 30, 40], "end": [10, 20, 30, 40, 50], "block_mean_sum": [1.0, 1.0, 1.0, 9.0, 11.0]})
    assert extreme_blocks(blocks, 10.0).start.tolist() == [40]
