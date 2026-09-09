import numpy as np

from src.pipeline import dcg_at_k, mrr_at_k, ndcg_at_k, success_at_k


def test_ideal_ranking_has_unit_ndcg() -> None:
    relevance = np.array([2, 1, 1, 0, 0])
    assert ndcg_at_k(relevance, 5) == 1.0


def test_better_order_improves_ndcg() -> None:
    weak = np.array([0, 1, 2, 0])
    strong = np.array([2, 1, 0, 0])
    assert dcg_at_k(strong, 4) > dcg_at_k(weak, 4)
    assert ndcg_at_k(strong, 4) > ndcg_at_k(weak, 4)


def test_mrr_and_success() -> None:
    relevance = np.array([0, 0, 1, 0])
    assert mrr_at_k(relevance, 4) == 1 / 3
    assert success_at_k(relevance, 3) == 1.0
    assert success_at_k(relevance, 2) == 0.0

