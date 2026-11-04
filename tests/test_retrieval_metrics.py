import math

import pytest

from app.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_counts_unique_relevant_results_in_prefix():
    assert recall_at_k(["a", "a", "b", "c"], {"a", "b", "d"}, 3) == pytest.approx(2 / 3)


def test_recall_at_k_is_zero_when_no_relevant_result_is_retrieved():
    assert recall_at_k(["x", "y"], {"a", "b"}, 2) == 0.0


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5


def test_reciprocal_rank_is_zero_when_no_relevant_result_is_retrieved():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k_matches_binary_relevance_definition():
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal_dcg = 1 + 1 / math.log2(3)

    assert ndcg_at_k(["x", "a", "y", "b"], {"a", "b"}, 4) == pytest.approx(expected_dcg / ideal_dcg)


def test_ndcg_does_not_count_duplicate_relevant_items_twice():
    assert ndcg_at_k(["a", "a"], {"a", "b"}, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))


@pytest.mark.parametrize("k", [0, -1])
def test_recall_and_ndcg_reject_nonpositive_k(k):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        recall_at_k([], {"a"}, k)
    with pytest.raises(ValueError, match="k must be greater than zero"):
        ndcg_at_k([], {"a"}, k)


def test_metrics_reject_empty_relevance_labels():
    with pytest.raises(ValueError, match="relevant_ids must contain"):
        recall_at_k(["a"], set(), 1)
    with pytest.raises(ValueError, match="relevant_ids must contain"):
        reciprocal_rank(["a"], set())
    with pytest.raises(ValueError, match="relevant_ids must contain"):
        ndcg_at_k(["a"], set(), 1)
