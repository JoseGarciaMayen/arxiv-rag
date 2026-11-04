"""Offline-labeled retrieval benchmark and PostgreSQL retrieval adapters."""

import hashlib
import re
from collections.abc import Callable
from math import fsum, log

from sqlalchemy import text

from app.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def stable_chunk_id(source: str, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{source}:{digest}"


def search_fts(question: str, engine, top_k: int) -> list[str]:
    statement = text("""
        SELECT source, content
        FROM benchmark_chunks
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
        FROM benchmark_chunks
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)
    with engine.connect() as connection:
        rows = connection.execute(statement, {"embedding": vector, "top_k": top_k})
        return [stable_chunk_id(row[0], row[1]) for row in rows]


def tokenize_bm25(text_value: str) -> list[str]:
    """Tokenize words and numbers for the local Okapi BM25 baseline."""
    return re.findall(r"[\w]+", text_value.casefold())


class BM25Index:
    """Deterministic BM25Okapi index over the frozen benchmark corpus."""

    def __init__(self, corpus_rows: list[dict], k1: float = 1.5, b: float = 0.75):
        if not corpus_rows:
            raise ValueError("BM25 corpus must contain at least one chunk")
        self.rows = corpus_rows
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize_bm25(row["content"]) for row in corpus_rows]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = fsum(self.lengths) / len(self.lengths)
        self.document_frequencies: dict[str, int] = {}
        for tokens in self.tokens:
            for token in set(tokens):
                self.document_frequencies[token] = self.document_frequencies.get(token, 0) + 1
        count = len(self.rows)
        self.idf = {
            token: log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in self.document_frequencies.items()
        }

    def search(self, query: str, top_k: int) -> list[str]:
        query_terms = tokenize_bm25(query)
        if not query_terms:
            return []
        scores = []
        for index, document in enumerate(self.tokens):
            frequencies: dict[str, int] = {}
            for term in document:
                frequencies[term] = frequencies.get(term, 0) + 1
            score = 0.0
            length_norm = self.k1 * (
                1 - self.b + self.b * self.lengths[index] / self.average_length
            )
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1) / (frequency + length_norm)
                    )
            if score > 0:
                row = self.rows[index]
                chunk_id = row.get("chunk_id") or stable_chunk_id(row["source"], row["content"])
                scores.append((score, index, chunk_id))
        scores.sort(key=lambda result: (-result[0], result[1]))
        return [chunk_id for _, _, chunk_id in scores[:top_k]]


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
                "retrieved_chunk_ids": retrieved,
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


def compare_retrievers(
    questions: list[dict],
    engine,
    embedder,
    corpus_rows: list[dict],
    top_k: int = 20,
) -> dict:
    bm25_index = BM25Index(corpus_rows)

    def fts(question: str) -> list[str]:
        return search_fts(question, engine, top_k)

    def dense(question: str) -> list[str]:
        return search_dense(question, engine, embedder, top_k)

    def bm25(question: str) -> list[str]:
        return bm25_index.search(question, top_k)

    return {
        "fts": evaluate_rankings(questions, fts),
        "bm25": evaluate_rankings(questions, bm25),
        "dense": evaluate_rankings(questions, dense),
        "hybrid_bm25_dense_rrf": evaluate_rankings(
            questions,
            lambda question: reciprocal_rank_fusion([dense(question), bm25(question)])[:top_k],
        ),
        "hybrid_rrf": evaluate_rankings(
            questions,
            lambda question: reciprocal_rank_fusion([dense(question), fts(question)])[:top_k],
        ),
    }
