"""Run the labeled benchmark against a PostgreSQL arxiv-rag corpus."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    from app.retrieval_benchmark import compare_retrievers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url", required=True, help="URL for the local benchmark PostgreSQL"
    )
    parser.add_argument("--top-k", type=int, default=20, help="Results per retriever (default: 20)")
    parser.add_argument("--output", type=Path, help="Write JSON results here instead of stdout")
    args = parser.parse_args()
    if args.top_k < 20:
        parser.error("--top-k must be at least 20 (the largest metric cutoff)")

    database_url = args.database_url
    if make_url(database_url).host not in {"localhost", "127.0.0.1", "::1", "db"}:
        parser.error(
            "benchmark runner only permits a local PostgreSQL host; remote DBs are not used"
        )

    corpus_path = ROOT / "benchmark" / "corpus.jsonl"
    manifest_path = ROOT / "benchmark" / "corpus-manifest.json"
    queries_path = ROOT / "benchmark" / "queries.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_bytes = corpus_path.read_bytes()
    corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()
    if corpus_hash != manifest["corpus_sha256"]:
        parser.error("corpus.jsonl does not match corpus-manifest.json; rebuild the corpus")

    corpus_rows = read_jsonl(corpus_path)
    corpus_ids = {row["chunk_id"] for row in corpus_rows}
    questions = read_jsonl(queries_path)
    unknown = sorted(
        {chunk_id for question in questions for chunk_id in question["relevant_chunk_ids"]}
        - corpus_ids
    )
    if unknown:
        parser.error(f"{len(unknown)} relevance labels do not exist in the frozen corpus")

    from app.models import EMBED_MODEL

    engine = create_engine(database_url)
    try:
        results = compare_retrievers(questions, engine, EMBED_MODEL, args.top_k)
    finally:
        engine.dispose()

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "corpus_sha256": corpus_hash,
        "questions_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "question_count": len(questions),
        "retrieval_depth": args.top_k,
        "methods": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote benchmark results to {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
