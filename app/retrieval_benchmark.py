"""Offline-labeled retrieval benchmark and PostgreSQL retrieval adapters."""

import hashlib
from collections.abc import Callable
from math import fsum

from sqlalchemy import text

from app.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def stable_chunk_id(source: str, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{source}:{digest}"


def search_fts(question: str, engine, top_k: int) -> list[str]:
    statement = text("""
        SELECT source, content
        FROM chunks
        WHERE content_tsv @@ websearch_to_tsquery('english', :query)
        ORDER BY ts_rank(content_tsv, websearch_to_tsquery('english', :query)) DESC
        LIMIT :top_k
    """)
    with engine.connect() as connection:
        rows = connection.execute(statement, {"query": question, "top_k": top_k})
        return [stable_chunk_id(row[0], row[1]) for row in rows]


def search_dense(question: str, engine, embedder, top_k: int) -> list[str]:
    vector = str(embedder.encode(question).tolist())
    statement = text("""
        SELECT source, content
        FROM chunks
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)
    with engine.connect() as connection:
        rows = connection.execute(statement, {"embedding": vector, "top_k": top_k})
        return [stable_chunk_id(row[0], row[1]) for row in rows]


def reciprocal_rank_fusion(rankings: list[list[str]], rrf_k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)


def evaluate_rankings(
    questions: list[dict],
    retrieve: Callable[[str], list[str]],
    cutoffs: tuple[int, ...] = (5, 10, 20),
) -> dict:
    per_query = []
    for question in questions:
        retrieved = retrieve(question["question"])
        relevant = question["relevant_chunk_ids"]
        per_query.append(
            {
                "query_id": question["query_id"],
                "reciprocal_rank": reciprocal_rank(retrieved, relevant),
                "cutoffs": {
                    str(k): {
                        "recall": recall_at_k(retrieved, relevant, k),
                        "ndcg": ndcg_at_k(retrieved, relevant, k),
                    }
                    for k in cutoffs
                },
            }
        )

    count = len(per_query)
    if not count:
        raise ValueError("Benchmark requires at least one labeled question")
    summary = {
        "mrr": fsum(row["reciprocal_rank"] for row in per_query) / count,
        "cutoffs": {
            str(k): {
                "recall": fsum(row["cutoffs"][str(k)]["recall"] for row in per_query) / count,
                "ndcg": fsum(row["cutoffs"][str(k)]["ndcg"] for row in per_query) / count,
            }
            for k in cutoffs
        },
    }
    return {"summary": summary, "per_query": per_query}


def compare_retrievers(questions: list[dict], engine, embedder, top_k: int = 20) -> dict:
    def fts(question: str) -> list[str]:
        return search_fts(question, engine, top_k)

    def dense(question: str) -> list[str]:
        return search_dense(question, engine, embedder, top_k)

    return {
        "fts": evaluate_rankings(questions, fts),
        "dense": evaluate_rankings(questions, dense),
        "hybrid_rrf": evaluate_rankings(
            questions,
            lambda question: reciprocal_rank_fusion([dense(question), fts(question)])[:top_k],
        ),
    }
