import numpy as np

from scripts.v0data.audit_rna_pseudobulk_depth import summarize_depths


def test_summarize_depths_applies_retained_group_mask() -> None:
    result = summarize_depths(
        ["a", "b", "c"],
        np.asarray([1e6, 1e7, 1e8]),
        np.asarray([2e6, 2e7, 2e8]),
        np.asarray([100, 1000, 10000]),
        {"b", "c"},
    )

    assert result["num_groups"] == 3
    assert result["retained_groups"] == 2
    assert result["retained_cell_quantiles"]["0.0"] == 1000
    assert result["retained_below_1000_cells"] == 0
    assert result["spearman_log_atac_vs_log_expression_depth"] == 1.0
