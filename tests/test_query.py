from unittest.mock import MagicMock

import pytest

from app.query import (
    build_system_message,
    query_with_context,
    reciprocal_rank_fusion,
    rerank,
    rewrite_query,
)


def _chunk(id: int, content: str = "text", source: str = "paper.pdf") -> dict:
    return {"id": id, "content": content, "source": source}


class TestReciprocalRankFusion:
    def test_doc_in_both_rankings_scores_highest(self):
        ranking1 = [_chunk(1), _chunk(2)]
        ranking2 = [_chunk(2), _chunk(3)]
        result = reciprocal_rank_fusion([ranking1, ranking2])
        assert result[0]["id"] == 2

    def test_preserves_order_for_single_ranking(self):
        ranking = [_chunk(1), _chunk(2), _chunk(3)]
        result = reciprocal_rank_fusion([ranking])
        assert [d["id"] for d in result] == [1, 2, 3]

    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fusion([]) == []

    def test_all_docs_present_in_output(self):
        ranking1 = [_chunk(1), _chunk(2)]
        ranking2 = [_chunk(3), _chunk(4)]
        result = reciprocal_rank_fusion([ranking1, ranking2])
        assert {d["id"] for d in result} == {1, 2, 3, 4}


class TestBuildSystemMessage:
    def test_includes_chunk_content(self):
        chunks = [_chunk(1, content="Attention is all you need")]
        msg = build_system_message(chunks)
        assert "Attention is all you need" in msg

    def test_includes_source_label(self):
        chunks = [_chunk(1, source="transformers.pdf")]
        msg = build_system_message(chunks)
        assert "transformers.pdf" in msg

    def test_lists_all_unique_sources(self):
        chunks = [
            _chunk(1, source="a.pdf"),
            _chunk(2, source="b.pdf"),
            _chunk(3, source="a.pdf"),
        ]
        msg = build_system_message(chunks)
        assert "a.pdf" in msg
        assert "b.pdf" in msg


class TestRerank:
    def test_orders_chunks_by_score_descending(self):
        from app.query import RERANK_MODEL

        RERANK_MODEL.predict.return_value = [0.1, 0.9, 0.5]
        chunks = [_chunk(1), _chunk(2), _chunk(3)]
        result = rerank("question", chunks)
        assert [d["id"] for d in result] == [2, 3, 1]

    def test_empty_input_returns_empty(self):
        assert rerank("question", []) == []


class TestRewriteQuery:
    def test_returns_original_question_when_no_history(self):
        result = rewrite_query("what is attention?", [])
        assert result == "what is attention?"

    def test_calls_configured_provider_and_returns_rewritten_query(self):
        import app.query as q

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="  rewritten query  "))]
        original_client = q.llm_client
        q.llm_client = MagicMock()
        q.llm_client.chat.completions.create.return_value = mock_resp

        history = [{"role": "user", "content": "tell me about transformers"}]
        result = rewrite_query("what about the attention mechanism?", history)

        assert result == "rewritten query"
        q.llm_client.chat.completions.create.assert_called_once()
        q.llm_client = original_client

    def test_raises_if_provider_returns_no_choices(self):
        import app.query as q

        mock_resp = MagicMock()
        mock_resp.choices = []
        original_client = q.llm_client
        q.llm_client = MagicMock()
        q.llm_client.chat.completions.create.return_value = mock_resp

        with pytest.raises(ValueError, match="groq returned no choices"):
            rewrite_query("question", [{"role": "user", "content": "prior"}])

        q.llm_client = original_client


class TestQueryWithContext:
    def test_returns_the_chunks_used_to_generate_the_answer(self, monkeypatch):
        import app.query as q

        engine = MagicMock()
        history = [{"role": "user", "content": "earlier question"}]
        chunks = [_chunk(1, content="Relevant paper passage")]
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="grounded answer"))]
        client = MagicMock()
        client.chat.completions.create.return_value = response

        monkeypatch.setattr(q, "rewrite_query", lambda question, history: "rewritten query")
        search = MagicMock(return_value=chunks)
        monkeypatch.setattr(q, "search_chunks", search)
        monkeypatch.setattr(q, "llm_client", client)

        answer, retrieved = query_with_context("original question", engine, history)

        assert answer == "grounded answer"
        assert retrieved is chunks
        search.assert_called_once_with("rewritten query", engine)
        request = client.chat.completions.create.call_args.kwargs
        assert request["messages"][0]["content"] == q.build_system_message(chunks)

    def test_no_results_returns_empty_context_without_generating(self, monkeypatch):
        import app.query as q

        client = MagicMock()
        monkeypatch.setattr(q, "rewrite_query", lambda question, history: question)
        monkeypatch.setattr(q, "search_chunks", MagicMock(return_value=[]))
        monkeypatch.setattr(q, "llm_client", client)

        answer, retrieved = query_with_context("unknown question", MagicMock())

        assert answer == (
            "I couldn't find relevant information in the indexed papers for that question."
        )
        assert retrieved == []
        client.chat.completions.create.assert_not_called()
