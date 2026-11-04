"""Load the frozen benchmark corpus into an isolated local PostgreSQL table."""

import argparse
import hashlib
import json
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True, help="URL for local PostgreSQL/pgvector")
    args = parser.parse_args()
    if make_url(args.database_url).host not in {"localhost", "127.0.0.1", "::1", "db"}:
        parser.error("benchmark loader only permits a local PostgreSQL host")

    manifest = json.loads((ROOT / "benchmark/corpus-manifest.json").read_text(encoding="utf-8"))
    corpus_path = ROOT / "benchmark/corpus.jsonl"
    if hashlib.sha256(corpus_path.read_bytes()).hexdigest() != manifest["corpus_sha256"]:
        parser.error("corpus.jsonl does not match corpus-manifest.json; rebuild the corpus")
    rows = [
        json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line
    ]

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(
        [row["content"] for row in rows],
        batch_size=32,
        show_progress_bar=True,
    )
    engine = create_engine(args.database_url)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(
                text("""
                CREATE TABLE IF NOT EXISTS benchmark_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(384) NOT NULL,
                    content_tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', content)
                    ) STORED
                )
            """)
            )
            connection.execute(
                text("""
                CREATE INDEX IF NOT EXISTS benchmark_chunks_tsv_idx
                ON benchmark_chunks USING GIN (content_tsv)
            """)
            )
            connection.execute(
                text("""
                CREATE INDEX IF NOT EXISTS benchmark_chunks_emb_idx
                ON benchmark_chunks USING hnsw (embedding vector_cosine_ops)
            """)
            )
            # This table materializes only the frozen benchmark corpus.
            # Refresh it fully so removed IDs cannot leak from an older version.
            connection.execute(text("TRUNCATE benchmark_chunks"))
            connection.execute(
                text("""
                    INSERT INTO benchmark_chunks (chunk_id, source, content, embedding)
                    VALUES (:chunk_id, :source, :content, CAST(:embedding AS vector))
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        source = EXCLUDED.source,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                """),
                [
                    {
                        "chunk_id": row["chunk_id"],
                        "source": row["source"],
                        "content": row["content"],
                        "embedding": str(embedding.tolist()),
                    }
                    for row, embedding in zip(rows, embeddings, strict=True)
                ],
            )
    finally:
        engine.dispose()

    print(f"Loaded {len(rows)} chunks into the isolated benchmark_chunks table.")


if __name__ == "__main__":
    main()
