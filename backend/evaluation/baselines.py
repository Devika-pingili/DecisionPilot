"""Deterministic baseline rankers restricted to an existing candidate set."""

from collections.abc import Iterable
from typing import Optional

import pandas as pd

from backend.features.history import HistoryStore


def _candidate_ids(candidate_product_ids: Iterable[int]) -> list[int]:
    ids = sorted(set(int(product_id) for product_id in candidate_product_ids))
    if not ids:
        return []
    return ids


def _rank(scores: dict[int, tuple], candidate_product_ids: Iterable[int]) -> list[int]:
    ids = _candidate_ids(candidate_product_ids)
    return sorted(ids, key=lambda product_id: (*scores.get(product_id, (0,)), product_id), reverse=False)


def _rank_desc(scores: dict[int, tuple], candidate_product_ids: Iterable[int]) -> list[int]:
    ids = _candidate_ids(candidate_product_ids)
    return sorted(
        ids,
        key=lambda product_id: tuple(-value for value in scores.get(product_id, (0,))) + (product_id,),
    )


def _history(
    history: HistoryStore,
    customer_id: int,
    target_order_id: int,
    target_order_number: int,
    history_context: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return history_context or history.history_for_target(
        customer_id, target_order_id, target_order_number
    )


def customer_frequency(
    history: HistoryStore,
    customer_id: int,
    target_order_id: int,
    target_order_number: int,
    candidate_product_ids: Iterable[int],
    *,
    history_context: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> list[int]:
    """Rank candidates by the customer's historical order count, then product ID."""
    _, transactions = _history(
        history, customer_id, target_order_id, target_order_number, history_context
    )
    counts = transactions.groupby("product_id")["order_id"].nunique()
    scores = {int(product_id): (int(count),) for product_id, count in counts.items()}
    return _rank_desc(scores, candidate_product_ids)


def customer_recency(
    history: HistoryStore,
    customer_id: int,
    target_order_id: int,
    target_order_number: int,
    candidate_product_ids: Iterable[int],
    *,
    history_context: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> list[int]:
    """Rank candidates by latest historical purchase order, then product ID."""
    _, transactions = _history(
        history, customer_id, target_order_id, target_order_number, history_context
    )
    latest = transactions.groupby("product_id")["order_number"].max()
    scores = {int(product_id): (int(order_number),) for product_id, order_number in latest.items()}
    return _rank_desc(scores, candidate_product_ids)


def previously_purchased(
    history: HistoryStore,
    customer_id: int,
    target_order_id: int,
    target_order_number: int,
    candidate_product_ids: Iterable[int],
    *,
    history_context: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> list[int]:
    """Rank by recency, then frequency, then historical reorder rate, then ID."""
    _, transactions = _history(
        history, customer_id, target_order_id, target_order_number, history_context
    )
    stats = transactions.groupby("product_id").agg(
        frequency=("order_id", "nunique"),
        latest_order_number=("order_number", "max"),
        reorder_count=("reordered", "sum"),
    )
    scores = {
        int(product_id): (
            int(row.latest_order_number),
            int(row.frequency),
            float(row.reorder_count / row.frequency),
        )
        for product_id, row in stats.iterrows()
    }
    return _rank_desc(scores, candidate_product_ids)


def global_popularity(
    history: HistoryStore,
    customer_id: int,
    target_order_id: int,
    target_order_number: int,
    candidate_product_ids: Iterable[int],
    *,
    history_context: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> list[int]:
    """Rank candidates by product purchases strictly before the target cutoff."""
    ids = _candidate_ids(candidate_product_ids)
    statistics = history.product_statistics(target_order_number, ids)
    scores = {
        int(row.product_id): (int(row.product_global_purchase_count),)
        for row in statistics.itertuples(index=False)
    }
    return _rank_desc(scores, ids)


def department_aware_popularity(
    history: HistoryStore,
    customer_id: int,
    target_order_id: int,
    target_order_number: int,
    candidate_product_ids: Iterable[int],
    *,
    history_context: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> list[int]:
    """Prefer product popularity in departments used by this customer historically.

    Within the active-department set, product purchase count is the ranking
    score. Products outside that set use the same cutoff-safe global score as a
    fallback, after all active-department candidates.
    """
    ids = _candidate_ids(candidate_product_ids)
    _, transactions = _history(
        history, customer_id, target_order_id, target_order_number, history_context
    )
    catalog = history.catalog()[["product_id", "department_id"]]
    historical = transactions.merge(catalog, on="product_id", how="left", validate="many_to_one")
    active_departments = set(historical["department_id"].dropna().astype(int))
    statistics = history.product_statistics(target_order_number, ids)
    counts = {
        int(row.product_id): int(row.product_global_purchase_count)
        for row in statistics.itertuples(index=False)
    }
    candidate_catalog = catalog.loc[catalog["product_id"].isin(ids)].set_index("product_id")
    scores = {
        product_id: (
            0 if int(candidate_catalog.loc[product_id, "department_id"]) in active_departments else 1,
            -counts.get(product_id, 0),
        )
        for product_id in ids
    }
    return sorted(ids, key=lambda product_id: (*scores[product_id], product_id))


BASELINES = {
    "customer_frequency": customer_frequency,
    "customer_recency": customer_recency,
    "previously_purchased": previously_purchased,
    "global_popularity": global_popularity,
    "department_aware_popularity": department_aware_popularity,
}
