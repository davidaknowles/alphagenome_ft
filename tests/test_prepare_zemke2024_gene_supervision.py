import json

import h5py
import numpy as np
import pandas as pd
import pytest

from scripts.v0data.zemke2024_rna_reprocessing.prepare_gene_supervision import (
    audit_molecule_discrepancy,
    retained_raw_feature_mask,
    target_groups_and_validity,
)
from scripts.v0data.zemke2024_rna_reprocessing.smoke_donor_aggregation import (
    smoke_donor,
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


def test_seurat_unique_suffix_maps_duplicate_raw_gene_names():
    mask = retained_raw_feature_mask(
        ("A", "B", "A", "C"),
        ("A", "A.1", "C"),
    )

    np.testing.assert_array_equal(mask, [True, False, True, True])


def test_zemke2024_donor_smoke_recovers_metadata_molecules(tmp_path):
    matrix_path = tmp_path / "donor.h5"
    with h5py.File(matrix_path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("barcodes", data=np.asarray([b"a", b"b"]))
        matrix.create_dataset("data", data=np.asarray([2, 3, 5]))
        matrix.create_dataset("indices", data=np.asarray([0, 1, 0]))
        matrix.create_dataset("indptr", data=np.asarray([0, 2, 3]))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray([b"ENSG1", b"ENSG2"]))
        features.create_dataset("name", data=np.asarray([b"G1", b"G2"]))
        features.create_dataset(
            "feature_type", data=np.asarray([b"Gene Expression", b"Gene Expression"])
        )
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame(
        {
            "bacrode": ["hc1_deep_a", "hc1_deep_b"],
            "orig.ident": ["hc1", "hc1"],
            "subclass": ["Astro", "Astro"],
            "nCount_RNA": [5, 5],
        }
    ).to_csv(metadata_path, sep="\t", index=False)
    targets_path = tmp_path / "targets.json"
    filtered_seurat_path = tmp_path / "filtered.h5"
    with h5py.File(filtered_seurat_path, "w") as handle:
        features = handle.create_group("assays").create_group("SCT")
        features.create_dataset("features", data=np.asarray([b"G1", b"G2"]))
    targets_path.write_text(
        json.dumps(
            {
                "heads": [
                    {
                        "kind": "rna_seq",
                        "targets": [
                            {"label": "Astro_all"},
                            {"label": "Astro1_all"},
                            {"label": "Astro2_all"},
                            {"label": "Micro1_all"},
                            {"label": "Micro2_all"},
                        ],
                    }
                ]
            }
        )
    )

    result = smoke_donor(
        donor="hc1",
        matrix_path=matrix_path,
        metadata_path=metadata_path,
        filtered_seurat_path=filtered_seurat_path,
        targets_path=targets_path,
    )

    assert result == {
        "donor": "hc1",
        "cells": 2,
        "filtered_raw_molecules": 10,
        "metadata_nCount_RNA": 10,
        "molecule_difference": 0,
        "relative_molecule_discrepancy": 0.0,
        "gene_features": 2,
        "valid_groups": 1,
        "nonempty_groups": 1,
    }


def test_molecule_discrepancy_enforces_relative_limit():
    with pytest.raises(ValueError, match="above the 0.1000% limit"):
        audit_molecule_discrepancy(
            np.asarray([1002]),
            np.asarray([1000]),
            donor="hc1",
            maximum_relative_discrepancy=0.001,
        )
