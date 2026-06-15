import pytest
from unittest.mock import MagicMock, patch
from app.query import reciprocal_rank_fusion, build_system_message, rerank, rewrite_query


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

    def test_calls_groq_and_returns_rewritten_query(self):
        import app.query as q
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="  rewritten query  "))]
        original_client = q.groq_client
        q.groq_client = MagicMock()
        q.groq_client.chat.completions.create.return_value = mock_resp

        history = [{"role": "user", "content": "tell me about transformers"}]
        result = rewrite_query("what about the attention mechanism?", history)

        assert result == "rewritten query"
        q.groq_client.chat.completions.create.assert_called_once()
        q.groq_client = original_client

    def test_raises_if_groq_returns_no_choices(self):
        import app.query as q
        mock_resp = MagicMock()
        mock_resp.choices = []
        original_client = q.groq_client
        q.groq_client = MagicMock()
        q.groq_client.chat.completions.create.return_value = mock_resp

        with pytest.raises(ValueError, match="Groq returned no choices"):
            rewrite_query("question", [{"role": "user", "content": "prior"}])

        q.groq_client = original_client
