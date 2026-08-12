import numpy as np

from scripts.v0data.zemke2024_rna_reprocessing.prepare_gene_supervision import (
    target_groups_and_validity,
)


def test_zemke2024_gene_validity_masks_only_unreleased_subtypes():
    labels = [
        "Astro_all",
        "Astro1_all",
        "Astro2_all",
        "Micro1_all",
        "Micro2_all",
        "Microglia_all",
        "SST_all",
    ]
    config = {
        "heads": [
            {
                "kind": "rna_seq",
                "targets": [{"label": label} for label in labels],
            }
        ]
    }

    groups, valid = target_groups_and_validity(config)

    assert groups == tuple(labels)
    np.testing.assert_array_equal(valid, [True, False, False, False, False, True, True])
