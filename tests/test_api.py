from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def test_app():
    from app.api import app

    return app


@pytest.fixture
def client(test_app):
    with patch("app.api.setup_db"):
        with TestClient(test_app) as c:
            yield c


def _mock_engine_connect(rows=None):
    """Return (context_manager, mock_conn) for patching engine.connect()."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value = rows or []
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_conn)
    cm.__exit__ = MagicMock(return_value=False)
    return cm, mock_conn


class TestUploadEndpoint:
    def test_rejects_non_pdf(self, client):
        res = client.post("/upload", files={"file": ("doc.txt", b"content", "text/plain")})
        assert res.status_code == 400
        assert "pdf" in res.json()["detail"].lower()

    def test_successful_upload_returns_chunk_count(self, client):
        with (
            patch("app.api.extract_text", return_value="raw text"),
            patch("app.api.clean_text", return_value="clean text"),
            patch("app.api.chunk_text", return_value=["chunk1", "chunk2", "chunk3"]),
            patch("app.api.embed_and_store"),
        ):
            res = client.post(
                "/upload",
                files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )
        assert res.status_code == 200
        assert res.json()["chunks"] == 3

    def test_corrupted_pdf_returns_422(self, client):
        with patch(
            "app.api.extract_text",
            side_effect=ValueError("Cannot read PDF: bad format"),
        ):
            res = client.post(
                "/upload",
                files={"file": ("bad.pdf", b"not a pdf", "application/pdf")},
            )
        assert res.status_code == 422
        assert "Cannot read PDF" in res.json()["detail"]


class TestQueryEndpoint:
    def test_empty_question_returns_400(self, client):
        res = client.post("/query", json={"question": "   "})
        assert res.status_code == 400

    def test_returns_answer(self, client):
        with patch("app.api.query", return_value="42 is the answer"):
            res = client.post("/query", json={"question": "What is the answer?"})
        assert res.status_code == 200
        assert res.json()["answer"] == "42 is the answer"

    def test_passes_history_to_query(self, client):
        history = [{"role": "user", "content": "prev question"}]
        with patch("app.api.query", return_value="answer") as mock_q:
            client.post("/query", json={"question": "follow up", "history": history})
        mock_q.assert_called_once_with("follow up", pytest.approx(mock_q.call_args[0][1]), history)


class TestDocumentsEndpoint:
    def test_returns_document_list(self, client):
        row = MagicMock()
        row.source = "paper.pdf"
        row.chunks = 42
        cm, _ = _mock_engine_connect(rows=[row])

        with patch("app.api.engine") as mock_engine:
            mock_engine.connect.return_value = cm
            res = client.get("/documents")

        assert res.status_code == 200
        assert res.json() == {"documents": [{"name": "paper.pdf", "chunks": 42}]}

    def test_returns_empty_list_when_no_documents(self, client):
        cm, _ = _mock_engine_connect(rows=[])
        with patch("app.api.engine") as mock_engine:
            mock_engine.connect.return_value = cm
            res = client.get("/documents")
        assert res.json() == {"documents": []}


class TestDeleteEndpoint:
    def test_deletes_document_successfully(self, client):
        cm, mock_conn = _mock_engine_connect()
        with patch("app.api.engine") as mock_engine:
            mock_engine.connect.return_value = cm
            res = client.delete("/document/paper.pdf")
        assert res.status_code == 200
        assert "paper.pdf" in res.json()["message"]
        mock_conn.commit.assert_called_once()

    def test_delete_calls_correct_sql(self, client):
        cm, mock_conn = _mock_engine_connect()
        with patch("app.api.engine") as mock_engine:
            mock_engine.connect.return_value = cm
            client.delete("/document/my_paper.pdf")
        call_args = mock_conn.execute.call_args
        assert call_args[0][1]["source"] == "my_paper.pdf"
