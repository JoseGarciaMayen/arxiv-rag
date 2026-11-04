import pytest

from app.retrieval_benchmark import (
    evaluate_rankings,
    reciprocal_rank_fusion,
    stable_chunk_id,
)


def test_stable_chunk_id_uses_source_and_content_hash():
    assert stable_chunk_id("paper.pdf", "passage") == (
        "paper.pdf:0886b051075687c18a9ae5075383feb8400fbe4328c6e4bd45b77b30f1e73b54"
    )


def test_reciprocal_rank_fusion_promotes_chunks_in_both_rankings():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[0] == "b"


def test_evaluate_rankings_reports_per_question_and_macro_averages():
    questions = [
        {"query_id": "one", "question": "q1", "relevant_chunk_ids": ["b"]},
        {"query_id": "two", "question": "q2", "relevant_chunk_ids": ["x"]},
    ]
    rankings = {"q1": ["b", "a"], "q2": ["a", "x"]}

    report = evaluate_rankings(questions, lambda question: rankings[question], cutoffs=(1, 2))

    assert report["summary"]["mrr"] == pytest.approx(0.75)
    assert report["summary"]["cutoffs"]["1"]["recall"] == pytest.approx(0.5)
    assert report["summary"]["cutoffs"]["2"]["recall"] == pytest.approx(1.0)
    assert [row["query_id"] for row in report["per_query"]] == ["one", "two"]


def test_evaluate_rankings_rejects_empty_question_set():
    with pytest.raises(ValueError, match="at least one labeled question"):
        evaluate_rankings([], lambda question: [])
