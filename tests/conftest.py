"""
Module-level mocks must run before any app.* import.
pytest loads conftest.py before collecting test files, so this is safe.
"""
import os
import sys
from unittest.mock import MagicMock
import numpy as np

# Required env vars — app.query raises RuntimeError at import if missing
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("GROQ_API_KEY", "test-key")

# Mock sentence_transformers before app.models is imported
MOCK_EMBED_MODEL = MagicMock()
MOCK_EMBED_MODEL.encode.return_value = np.zeros((1, 384), dtype=np.float32)

MOCK_RERANK_MODEL = MagicMock()
MOCK_RERANK_MODEL.predict.return_value = np.array([0.9])

_mock_st = MagicMock()
_mock_st.SentenceTransformer.return_value = MOCK_EMBED_MODEL
_mock_st.CrossEncoder.return_value = MOCK_RERANK_MODEL
sys.modules["sentence_transformers"] = _mock_st

# Mock app.models so ingest.py and query.py share the same instances
_mock_models = MagicMock()
_mock_models.EMBED_MODEL = MOCK_EMBED_MODEL
_mock_models.RERANK_MODEL = MOCK_RERANK_MODEL
sys.modules["app.models"] = _mock_models
