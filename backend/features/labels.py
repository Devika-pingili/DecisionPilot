"""Target-label construction kept separate from feature-history construction."""

from typing import Any

import pandas as pd

from .candidates import generate_candidates
from .history import HistoryStore


PREDICTION_KEYS = [
    "customer_id",
    "target_order_id",
    "target_order_number",
    "candidate_product_id",
]


def _require_prediction_keys(frame: pd.DataFrame, name: str) -> None:
    missing = [column for column in PREDICTION_KEYS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    if frame.duplicated(PREDICTION_KEYS).any():
        raise ValueError(f"{name} contains duplicate prediction rows")


def build_labels(
    prediction_rows: pd.DataFrame,
    target_products: pd.DataFrame,
) -> pd.DataFrame:
    """Attach binary product-presence labels to candidate prediction rows.

    ``target_products`` is supplied by the caller. This function does not load
    any file, so feature construction cannot accidentally read train targets.
    """
    _require_prediction_keys(prediction_rows, "prediction_rows")
    required = ["order_id", "product_id"]
    missing = [column for column in required if column not in target_products.columns]
    if missing:
        raise ValueError(f"target_products is missing required columns: {', '.join(missing)}")
    if target_products.duplicated(required).any():
        raise ValueError("target_products contains duplicate (order_id, product_id) rows")

    positives = target_products[required].drop_duplicates().rename(
        columns={"order_id": "target_order_id", "product_id": "candidate_product_id"}
    )
    result = prediction_rows.merge(
        positives.assign(y=1),
        on=["target_order_id", "candidate_product_id"],
        how="left",
        validate="one_to_one",
    )
    result["y"] = result["y"].fillna(0).astype("int8")
    return result[PREDICTION_KEYS + ["y"]]


def build_rolling_training_examples(
    history: HistoryStore,
    *,
    recent_order_count: int = 3,
    candidate_limit: int = 200,
    source_limit: int = 100,
) -> pd.DataFrame:
    """Build prior-only rolling labels using each prior order as a target.

    For orders 1..4, this yields histories before orders 2, 3, and 4. The
    target rows are read only to create labels, never passed to features.
    """
    rows: list[pd.DataFrame] = []
    for customer_id, customer_orders in history.orders.loc[
        history.orders["eval_set"] == "prior"
    ].groupby("user_id", sort=True):
        customer_orders = customer_orders.sort_values(["order_number", "order_id"])
        for target in customer_orders.iloc[1:].itertuples(index=False):
            candidates = generate_candidates(
                history,
                int(customer_id),
                int(target.order_id),
                int(target.order_number),
                recent_order_count=recent_order_count,
                candidate_limit=candidate_limit,
                source_limit=source_limit,
            )
            target_products = history.target_products(int(target.order_id))
            rows.append(build_labels(candidates, target_products))

    if not rows:
        return pd.DataFrame(columns=PREDICTION_KEYS + ["y"])
    return pd.concat(rows, ignore_index=True).sort_values(PREDICTION_KEYS).reset_index(drop=True)
