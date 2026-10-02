"""Deterministic sparse candidate generation and recall diagnostics."""

from typing import Optional

import pandas as pd

from .history import HistoryStore


PREDICTION_KEYS = [
    "customer_id",
    "target_order_id",
    "target_order_number",
    "candidate_product_id",
]


def _ranked_products(
    frame: pd.DataFrame,
    product_column: str = "product_id",
    limit: Optional[int] = None,
) -> list[int]:
    if frame.empty:
        return []
    ranked = frame.sort_values(
        ["score", product_column], ascending=[False, True]
    )[product_column].astype(int).tolist()
    return ranked if limit is None else ranked[:limit]


def generate_candidates(
    history: HistoryStore,
    customer_id: int,
    target_order_id: int,
    target_order_number: int,
    *,
    recent_order_count: int = 3,
    candidate_limit: int = 200,
    source_limit: int = 100,
    minimum_reorder_purchases: int = 2,
    minimum_reorder_rate: float = 0.5,
) -> pd.DataFrame:
    """Generate a deterministic union of sparse, historical product candidates."""
    if recent_order_count < 1 or candidate_limit < 1 or source_limit < 1:
        raise ValueError("candidate limits and recent_order_count must be positive")

    historical_orders, historical_transactions = history.history_for_target(
        customer_id, target_order_id, target_order_number
    )
    catalog = history.catalog()
    source_products: list[int] = []

    customer_counts = historical_transactions.groupby("product_id").size().rename("score")
    source_products.extend(
        _ranked_products(
            customer_counts.reset_index(), limit=source_limit
        )
    )

    recent_order_ids = historical_orders.tail(recent_order_count)["order_id"]
    recent = historical_transactions.loc[
        historical_transactions["order_id"].isin(set(recent_order_ids))
    ].groupby("product_id").size().rename("score")
    source_products.extend(_ranked_products(recent.reset_index(), limit=source_limit))

    source_products.extend(
        _ranked_products(customer_counts.reset_index(), limit=source_limit)
    )

    customer_reorders = historical_transactions.groupby("product_id").agg(
        purchase_count=("order_id", "size"),
        reorder_count=("reordered", "sum"),
    )
    customer_reorders["score"] = (
        customer_reorders["reorder_count"] / customer_reorders["purchase_count"]
    )
    strong_reorders = customer_reorders.reset_index().loc[
        (customer_reorders.reset_index()["purchase_count"] >= minimum_reorder_purchases)
        & (customer_reorders.reset_index()["score"] >= minimum_reorder_rate)
    ]
    source_products.extend(_ranked_products(strong_reorders, limit=source_limit))

    catalog_lookup = catalog[["product_id", "aisle_id", "department_id"]]
    customer_catalog_history = historical_transactions.merge(
        catalog_lookup, on="product_id", how="left", validate="many_to_one"
    )
    if customer_catalog_history[["aisle_id", "department_id"]].isna().any().any():
        raise ValueError("historical products are missing catalog relationships")
    departments = set(customer_catalog_history["department_id"])
    aisles = set(customer_catalog_history["aisle_id"])
    if history.is_indexed:
        category_popular = history.top_products(
            target_order_number, source_limit, departments, aisles
        )
        global_counts = history.top_products(target_order_number, source_limit)
    else:
        as_of_transactions = history.transactions_before_order_number(target_order_number)
        global_counts = as_of_transactions.merge(
            catalog_lookup, on="product_id", how="left", validate="many_to_one"
        ).groupby("product_id").size().rename("score").reset_index()
        catalog_popular = global_counts.merge(catalog_lookup, on="product_id", how="inner")
        category_popular = catalog_popular.loc[
            catalog_popular["department_id"].isin(departments)
            | catalog_popular["aisle_id"].isin(aisles)
        ]
    source_products.extend(_ranked_products(category_popular, limit=source_limit))

    source_products.extend(_ranked_products(global_counts, limit=source_limit))

    seen: set[int] = set()
    ordered_products: list[int] = []
    for product_id in source_products:
        product_id = int(product_id)
        if product_id not in seen:
            seen.add(product_id)
            ordered_products.append(product_id)
        if len(ordered_products) >= candidate_limit:
            break

    result = pd.DataFrame(
        {
            "customer_id": int(customer_id),
            "target_order_id": int(target_order_id),
            "target_order_number": int(target_order_number),
            "candidate_product_id": ordered_products,
        }
    )
    if result.empty:
        return pd.DataFrame(columns=PREDICTION_KEYS)
    return result[PREDICTION_KEYS]


def measure_candidate_recall(
    candidate_rows: pd.DataFrame,
    target_products: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return per-target and aggregate candidate recall diagnostics."""
    required_candidates = ["customer_id", "target_order_id", "candidate_product_id"]
    if any(column not in candidate_rows.columns for column in required_candidates):
        raise ValueError("candidate_rows is missing candidate recall columns")
    if not {"order_id", "product_id"}.issubset(target_products.columns):
        raise ValueError("target_products requires order_id and product_id")
    if candidate_rows.duplicated(
        ["customer_id", "target_order_id", "candidate_product_id"]
    ).any():
        raise ValueError("candidate_rows contains duplicate candidate rows")
    if target_products.duplicated(["order_id", "product_id"]).any():
        raise ValueError("target_products contains duplicate target products")

    candidate_groups = candidate_rows.groupby("target_order_id")
    diagnostics: list[dict[str, float | int]] = []
    for order_id, target_group in target_products.groupby("order_id", sort=True):
        target_set = set(target_group["product_id"].astype(int))
        candidate_set = set()
        customer_id = None
        target_order_number = None
        if order_id in candidate_groups.groups:
            candidates = candidate_groups.get_group(order_id)
            candidate_set = set(candidates["candidate_product_id"].astype(int))
            customer_id = int(candidates.iloc[0]["customer_id"])
            if "target_order_number" in candidates:
                target_order_number = int(candidates.iloc[0]["target_order_number"])
        matched = target_set & candidate_set
        diagnostics.append(
            {
                "customer_id": customer_id,
                "target_order_id": int(order_id),
                "target_order_number": target_order_number,
                "target_product_count": len(target_set),
                "candidate_count": len(candidate_set),
                "matched_target_products": len(matched),
                "missed_target_products": len(target_set - candidate_set),
                "candidate_recall": len(matched) / len(target_set) if target_set else 0.0,
            }
        )

    per_target = pd.DataFrame(diagnostics)
    aggregate = {
        "mean_candidate_recall": float(per_target["candidate_recall"].mean())
        if not per_target.empty
        else 0.0,
        "mean_candidate_count": float(per_target["candidate_count"].mean())
        if not per_target.empty
        else 0.0,
        "min_candidate_count": float(per_target["candidate_count"].min())
        if not per_target.empty
        else 0.0,
        "max_candidate_count": float(per_target["candidate_count"].max())
        if not per_target.empty
        else 0.0,
        "target_order_count": float(len(per_target)),
        "orders_with_zero_candidates": float(
            (per_target["candidate_count"] == 0).sum()
        )
        if not per_target.empty
        else 0.0,
    }
    return per_target, aggregate
