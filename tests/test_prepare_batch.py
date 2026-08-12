import numpy as np

from alphagenome_ft.finetune.data import prepare_batch


def test_prepare_batch_allows_gene_only_head_without_coverage_targets():
    batch = {
        "sequences": np.zeros((2, 4, 4), dtype=np.float32),
        "negative_strand_mask": np.asarray([False, True]),
        "targets_atac": np.ones((2, 4, 1), dtype=np.float32),
        "gene_weights_rna": np.ones((2, 1, 1), dtype=np.float32),
        "gene_targets_rna": np.ones((2, 1, 1), dtype=np.float32),
        "gene_strands_rna": np.zeros((2, 1), dtype=np.int8),
        "gene_valid_rna": np.ones((2, 1), dtype=bool),
    }

    prepared = prepare_batch(batch, organism_index_value=0, head_names=("atac", "rna"))

    assert "targets_atac" in prepared
    assert "targets_rna" not in prepared
    assert "gene_targets_rna" in prepared
