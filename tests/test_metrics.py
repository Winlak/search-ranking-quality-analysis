import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    dcg_at_k,
    generate,
    make_candidate_order,
    mrr_at_k,
    ndcg_at_k,
    shared_judgments,
    success_at_k,
    validate_sql_parity,
)


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


def test_empty_relevance_cannot_become_relevant() -> None:
    relevance = np.zeros(10, dtype=int)
    for seed in range(100):
        order = make_candidate_order(relevance, np.random.default_rng(seed), seed % 2 == 0)
        assert sorted(order) == list(range(10))
        assert ndcg_at_k(relevance[order]) == 0
        assert mrr_at_k(relevance[order]) == 0


def test_generator_is_reproducible_and_judgments_are_shared() -> None:
    metrics, rows = generate(queries=40)
    other_metrics, other_rows = generate(queries=40)
    pd.testing.assert_frame_equal(metrics, other_metrics)
    pd.testing.assert_frame_equal(rows, other_rows)
    assert len(shared_judgments(rows)) == 400
    assert (metrics["baseline_zero_relevant10"] == metrics["candidate_zero_relevant10"]).all()


def test_sql_matches_python_per_query() -> None:
    metrics, rows = generate(queries=200)
    result = validate_sql_parity(metrics, rows)
    assert result["status"] == "passed"
    assert result["max_absolute_error"] < 1e-12


def test_changed_judgment_is_rejected() -> None:
    _, rows = generate(queries=4)
    row = rows.index[rows["ranker"] == "candidate"][0]
    rows.loc[row, "relevance"] = (rows.loc[row, "relevance"] + 1) % 3
    with pytest.raises(ValueError, match="immutable"):
        shared_judgments(rows)


def test_duplicate_document_is_rejected() -> None:
    _, rows = generate(queries=4)
    rows.loc[1, "doc_id"] = rows.loc[0, "doc_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        shared_judgments(rows)


def test_sql_parity_detects_incorrect_python_metric() -> None:
    metrics, rows = generate(queries=5)
    metrics.loc[0, "baseline_ndcg10"] += 0.05
    with pytest.raises(AssertionError, match="mismatch"):
        validate_sql_parity(metrics, rows)
