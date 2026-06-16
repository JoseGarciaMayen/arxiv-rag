import os
import re
from collections import Counter
from pathlib import Path

import fitz  # pymupdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text

from app.models import EMBED_MODEL

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# Cleaning
_TRAILING_SECTION_RE = re.compile(
    r"\n\s*(?:references|bibliography|acknowledgements?|appendix)\s*\n",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_EXCESS_NEWLINE_RE = re.compile(r"\n{3,}")

# First-line patterns that signal a noisy/header chunk
_NOISY_FIRST_LINE_RE = re.compile(
    r"^(?:"
    r"published\s+(?:as\s+)?a?\s*(?:conference|journal|workshop)"
    r"|under\s+review\s+as"
    r"|preprint\b"
    r"|proceedings\s+of"
    r"|\[\d[\d,\s]*\]\s+\w"
    r"|\d+$"
    r")",
    re.IGNORECASE,
)


def setup_db(engine):
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                source TEXT,
                content TEXT,
                embedding vector(384),
                content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            )
        """)
        )
        conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (content_tsv)
        """)
        )
        conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks
            USING hnsw (embedding vector_cosine_ops)
        """)
        )
        conn.commit()


def extract_text(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Cannot read PDF: {e}") from e
    pages = [page.get_text() for page in doc]

    threshold = max(3, len(pages) * 0.30)
    line_counts: Counter = Counter()
    for page_text in pages:
        for line in set(page_text.splitlines()):
            if line.strip():
                line_counts[line.strip()] += 1
    running_lines = {line for line, count in line_counts.items() if count >= threshold}

    cleaned: list[str] = []
    for page_text in pages:
        lines = [line for line in page_text.splitlines() if line.strip() not in running_lines]
        cleaned.append("\n".join(lines))

    return "\n".join(cleaned)


def clean_text(raw: str) -> str:
    raw = raw.replace("\x00", "")
    match = _TRAILING_SECTION_RE.search(raw)
    if match:
        raw = raw[: match.start()]
    raw = _HYPHEN_BREAK_RE.sub(r"\1\2", raw)
    raw = _URL_RE.sub("", raw)
    raw = _EXCESS_NEWLINE_RE.sub("\n\n", raw)

    return raw.strip()


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
    )
    return splitter.split_text(text)


def is_valid_chunk(chunk: str) -> bool:
    stripped = chunk.strip()

    if len(stripped) < 100:
        return False

    first_line = stripped.split("\n")[0].strip()
    if _NOISY_FIRST_LINE_RE.match(first_line):
        return False

    words = stripped.split()
    if len(words) < 15:
        return False

    # Reject chunks dominated by citation markers like [1], [2, 3], [12]
    citation_hits = len(re.findall(r"\[\d[\d,\s]*\]", stripped))
    if citation_hits >= 5:
        return False

    # Reject chunks where less than half the characters are alphanumeric
    alnum_ratio = sum(c.isalnum() or c.isspace() for c in stripped) / len(stripped)
    if alnum_ratio < 0.5:
        return False

    # Reject chunks with very short average word length
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 3:
        return False

    return True


def embed_and_store(chunks: list[str], source: str, engine) -> int:
    if engine is None:
        engine = create_engine(DB_URL)
    clean_source = source.replace("\x00", "")
    valid_chunks = [c for c in chunks if is_valid_chunk(c)]
    embeddings = EMBED_MODEL.encode(valid_chunks, show_progress_bar=True)
    rows = [
        {
            "source": clean_source,
            "content": chunk.replace("\x00", ""),
            "embedding": str(embedding.tolist()),
        }
        for chunk, embedding in zip(valid_chunks, embeddings, strict=False)
    ]
    with engine.connect() as conn:
        # Re-ingesting the same source replaces its chunks instead of duplicating them
        conn.execute(
            text("DELETE FROM chunks WHERE source = :source"),
            {"source": clean_source},
        )
        if rows:
            conn.execute(
                text(
                    "INSERT INTO chunks (source, content, embedding) "
                    "VALUES (:source, :content, :embedding)"
                ),
                rows,
            )
        conn.commit()
    return len(rows)


def ingest_text(pdf_path: str):
    engine = create_engine(DB_URL)
    setup_db(engine)
    print(f"Extracting text from {pdf_path}...")
    raw_text = extract_text(pdf_path)
    print(f"Extracted text: {len(raw_text)} characters")
    clean = clean_text(raw_text)
    print(f"Cleaned text: {len(clean)} characters")
    chunks = chunk_text(clean)
    print(f"Chunks generated: {len(chunks)}")
    embed_and_store(chunks, source=Path(pdf_path).name, engine=engine)
    print("Ingestion complete.")


if __name__ == "__main__":
    import sys

    ingest_text(sys.argv[1])
