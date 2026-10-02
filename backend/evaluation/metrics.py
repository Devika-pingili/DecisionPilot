"""Per-target ranking metrics for binary product-presence recommendations."""

import math
from typing import Iterable


def _top_k(recommended: Iterable[int], k: int) -> list[int]:
    if k < 1:
        raise ValueError("k must be positive")
    return list(recommended)[:k]


def precision_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    top = _top_k(recommended, k)
    if not top:
        return 0.0
    return sum(product_id in relevant for product_id in top) / len(top)


def recall_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(product_id in relevant for product_id in _top_k(recommended, k)) / len(relevant)


def f1_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    precision = precision_at_k(recommended, relevant, k)
    recall = recall_at_k(recommended, relevant, k)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def average_precision_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = _top_k(recommended, k)
    hits = 0
    score = 0.0
    for rank, product_id in enumerate(top, start=1):
        if product_id in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, product_id in enumerate(_top_k(recommended, k), start=1)
        if product_id in relevant
    )
    ideal_count = min(len(relevant), k)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def ranking_metrics(
    recommended: Iterable[int],
    relevant: Iterable[int],
    k_values: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """Calculate standard per-target metrics for each requested K."""
    recommended = list(recommended)
    relevant_set = set(int(product_id) for product_id in relevant)
    result: dict[str, float] = {}
    for k in k_values:
        if k < 1:
            raise ValueError("k values must be positive")
        result[f"precision_at_{k}"] = precision_at_k(recommended, relevant_set, k)
        result[f"recall_at_{k}"] = recall_at_k(recommended, relevant_set, k)
        result[f"f1_at_{k}"] = f1_at_k(recommended, relevant_set, k)
        result[f"map_at_{k}"] = average_precision_at_k(recommended, relevant_set, k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(recommended, relevant_set, k)
    return result
