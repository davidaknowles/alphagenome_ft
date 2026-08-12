import numpy as np
import pytest

from alphagenome_ft.finetune.data import _balance_gene_window_order


def _batches(order: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [order[index : index + batch_size] for index in range(0, len(order), batch_size)]


def test_balanced_order_preserves_windows_and_spreads_gene_supervision() -> None:
    order = np.asarray([7, 2, 10, 0, 5, 1, 8, 3, 11, 4, 9, 6])
    gene_counts = np.asarray([0, 4, 0, 0, 3, 0, 0, 2, 0, 0, 1, 0])

    balanced = _balance_gene_window_order(
        order, gene_counts, batch_size=3, drop_last=False
    )

    np.testing.assert_array_equal(np.sort(balanced), np.arange(12))
    assert all(np.sum(gene_counts[batch]) > 0 for batch in _batches(balanced, 3))


def test_balanced_order_uses_each_selected_drop_last_window_once() -> None:
    order = np.asarray([6, 3, 2, 5, 1, 0, 4])
    gene_counts = np.asarray([0, 1, 0, 2, 0, 0, 3])

    balanced = _balance_gene_window_order(
        order, gene_counts, batch_size=3, drop_last=True
    )

    assert len(balanced) == 6
    assert set(balanced) == set(order[:6])
    assert len(set(balanced)) == len(balanced)


def test_balanced_order_retains_partial_final_batch() -> None:
    order = np.arange(10)
    gene_counts = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 0])

    balanced = _balance_gene_window_order(
        order, gene_counts, batch_size=4, drop_last=False
    )

    assert [len(batch) for batch in _batches(balanced, 4)] == [4, 4, 2]
    np.testing.assert_array_equal(np.sort(balanced), order)


def test_balanced_order_rejects_invalid_gene_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _balance_gene_window_order(
            np.asarray([0, 1]), np.asarray([1, -1]), batch_size=2, drop_last=False
        )
