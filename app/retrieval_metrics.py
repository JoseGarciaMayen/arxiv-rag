"""Pure ranking metrics for retrieval evaluation."""

from collections.abc import Iterable
from math import log2
from typing import TypeVar

Item = TypeVar("Item")


def _validate_inputs(relevant_ids: Iterable[Item], k: int) -> set[Item]:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must contain at least one item")
    return relevant


def recall_at_k(retrieved_ids: Iterable[Item], relevant_ids: Iterable[Item], k: int) -> float:
    """Return the fraction of relevant items present in the top-k results."""
    relevant = _validate_inputs(relevant_ids, k)
    retrieved = set()
    for index, item in enumerate(retrieved_ids):
        if index >= k:
            break
        retrieved.add(item)
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: Iterable[Item], relevant_ids: Iterable[Item]) -> float:
    """Return the reciprocal rank of the first relevant result, or zero if absent."""
    relevant = _validate_inputs(relevant_ids, 1)
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Iterable[Item], relevant_ids: Iterable[Item], k: int) -> float:
    """Return binary-relevance nDCG@k; duplicate results do not earn extra gain."""
    relevant = _validate_inputs(relevant_ids, k)
    dcg = 0.0
    seen_relevant = set()
    for rank, item in enumerate(retrieved_ids):
        if rank >= k:
            break
        if item in relevant and item not in seen_relevant:
            dcg += 1.0 / log2(rank + 2)
            seen_relevant.add(item)

    ideal_hits = min(k, len(relevant))
    ideal_dcg = sum(1.0 / log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal_dcg
