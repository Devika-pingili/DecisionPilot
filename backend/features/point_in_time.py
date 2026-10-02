"""Point-in-time personalized features for sparse candidate rows."""

from typing import Iterable

import pandas as pd

from .history import HistoryStore


PREDICTION_KEYS = [
    "customer_id",
    "target_order_id",
    "target_order_number",
    "candidate_product_id",
]

FEATURE_COLUMNS = PREDICTION_KEYS + [
    "product_id",
    "product_name",
    "aisle_id",
    "department_id",
    "customer_product_purchase_count",
    "customer_product_reorder_count",
    "customer_product_reorder_rate",
    "orders_since_last_product_purchase",
    "days_since_last_product_purchase",
    "customer_product_frequency",
    "customer_recent_product_flag",
    "customer_total_orders",
    "customer_average_basket_size",
    "customer_recent_basket_size",
    "product_global_purchase_count",
    "product_global_reorder_count",
    "product_historical_reorder_rate",
    "department_purchase_count",
    "aisle_purchase_count",
    "customer_department_affinity",
    "customer_aisle_affinity",
]


def _require_prediction_rows(rows: pd.DataFrame) -> None:
    missing = [column for column in PREDICTION_KEYS if column not in rows.columns]
    if missing:
        raise ValueError(f"prediction_rows is missing required columns: {', '.join(missing)}")
    if rows.duplicated(PREDICTION_KEYS).any():
        raise ValueError("prediction_rows contains duplicate prediction rows")


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float).div(denominator.astype(float))
    return result.where(denominator != 0)


def _days_since_last_purchase(
    historical_orders: pd.DataFrame,
    last_order_number: int,
    target_order_number: int,
) -> float:
    """Sum known intervals after the product purchase and before the target.

    ``days_since_prior_order`` on order N describes the interval ending at N.
    It is therefore excluded, as requested; only intervals on orders M+1..N-1
    are included after a product's last purchase on order M.
    """
    intervals = historical_orders.loc[
        (historical_orders["order_number"] > last_order_number)
        & (historical_orders["order_number"] < target_order_number),
        "days_since_prior_order",
    ]
    if intervals.empty:
        return 0.0
    if intervals.isna().any():
        return float("nan")
    return float(intervals.sum())


def _product_catalog(history: HistoryStore, product_ids: Iterable[int]) -> pd.DataFrame:
    catalog = history.catalog()
    result = catalog.loc[catalog["product_id"].isin(set(product_ids))].copy()
    if len(result) != len(set(product_ids)):
        missing = sorted(set(product_ids) - set(result["product_id"]))
        raise ValueError(f"candidate products missing from catalog: {missing}")
    if result[["aisle_id", "department_id"]].isna().any().any():
        raise ValueError("candidate products have missing catalog relationships")
    return result


def build_point_in_time_features(
    history: HistoryStore,
    prediction_rows: pd.DataFrame,
    *,
    recent_order_count: int = 3,
) -> pd.DataFrame:
    """Build features for candidate rows using only history before each target."""
    _require_prediction_rows(prediction_rows)
    if recent_order_count < 1:
        raise ValueError("recent_order_count must be positive")
    if prediction_rows.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    output: list[pd.DataFrame] = []
    for (customer_id, target_order_id, target_order_number), group in prediction_rows.groupby(
        ["customer_id", "target_order_id", "target_order_number"], sort=True
    ):
        historical_orders, historical_transactions = history.history_for_target(
            int(customer_id), int(target_order_id), int(target_order_number)
        )
        candidate_ids = group["candidate_product_id"].astype(int).tolist()
        catalog = _product_catalog(history, candidate_ids)
        candidate_frame = group[PREDICTION_KEYS].copy()
        candidate_frame["candidate_product_id"] = candidate_frame["candidate_product_id"].astype(int)
        candidate_frame = candidate_frame.merge(
            catalog[["product_id", "product_name", "aisle_id", "department_id"]],
            left_on="candidate_product_id",
            right_on="product_id",
            how="left",
            validate="one_to_one",
        )

        product_history = historical_transactions.groupby("product_id").agg(
            customer_product_purchase_count=("order_id", "nunique"),
            customer_product_reorder_count=("reordered", "sum"),
            last_product_order_number=("order_number", "max"),
        ).reset_index()
        candidate_frame = candidate_frame.merge(
            product_history.rename(columns={"product_id": "candidate_product_id"}),
            on="candidate_product_id",
            how="left",
            validate="one_to_one",
        )
        candidate_frame["customer_product_purchase_count"] = candidate_frame[
            "customer_product_purchase_count"
        ].fillna(0).astype("int64")
        candidate_frame["customer_product_reorder_count"] = candidate_frame[
            "customer_product_reorder_count"
        ].fillna(0).astype("int64")
        candidate_frame["customer_product_reorder_rate"] = _safe_rate(
            candidate_frame["customer_product_reorder_count"],
            candidate_frame["customer_product_purchase_count"],
        )
        candidate_frame["orders_since_last_product_purchase"] = (
            int(target_order_number) - candidate_frame["last_product_order_number"] - 1
        )
        candidate_frame.loc[
            candidate_frame["last_product_order_number"].isna(),
            "orders_since_last_product_purchase",
        ] = pd.NA
        candidate_frame["orders_since_last_product_purchase"] = candidate_frame[
            "orders_since_last_product_purchase"
        ].astype("Int64")

        days_by_product = {}
        for product_id, last_order_number in product_history[
            ["product_id", "last_product_order_number"]
        ].itertuples(index=False):
            days_by_product[int(product_id)] = _days_since_last_purchase(
                historical_orders, int(last_order_number), int(target_order_number)
            )
        candidate_frame["days_since_last_product_purchase"] = candidate_frame[
            "candidate_product_id"
        ].map(days_by_product)

        customer_order_count = len(historical_orders)
        basket_sizes = historical_transactions.groupby("order_id").size()
        candidate_frame["customer_product_frequency"] = (
            candidate_frame["customer_product_purchase_count"] / customer_order_count
            if customer_order_count
            else 0.0
        )
        recent_order_ids = set(historical_orders.tail(recent_order_count)["order_id"])
        recent_products = set(
            historical_transactions.loc[
                historical_transactions["order_id"].isin(recent_order_ids), "product_id"
            ]
        )
        candidate_frame["customer_recent_product_flag"] = candidate_frame[
            "candidate_product_id"
        ].isin(recent_products).astype("int8")
        candidate_frame["customer_total_orders"] = customer_order_count
        candidate_frame["customer_average_basket_size"] = (
            float(basket_sizes.mean()) if not basket_sizes.empty else float("nan")
        )
        candidate_frame["customer_recent_basket_size"] = (
            float(basket_sizes.loc[historical_orders.iloc[-1]["order_id"]])
            if not historical_orders.empty
            and historical_orders.iloc[-1]["order_id"] in basket_sizes
            else float("nan")
        )

        if history.is_indexed:
            global_product = history.product_statistics(
                int(target_order_number), candidate_ids
            )
            category_statistics = history.category_statistics(int(target_order_number))
        else:
            global_transactions = history.transactions_before_order_number(
                int(target_order_number)
            )
            global_product = global_transactions.groupby("product_id").agg(
                product_global_purchase_count=("order_id", "size"),
                product_global_reorder_count=("reordered", "sum"),
            ).reset_index()
            category_statistics = None
        global_product = global_product[[
            "product_id", "product_global_purchase_count", "product_global_reorder_count"
        ]]
        candidate_frame = candidate_frame.merge(
            global_product.rename(columns={"product_id": "candidate_product_id"}),
            on="candidate_product_id",
            how="left",
            validate="one_to_one",
        )
        for column in ["product_global_purchase_count", "product_global_reorder_count"]:
            candidate_frame[column] = candidate_frame[column].fillna(0).astype("int64")
        candidate_frame["product_historical_reorder_rate"] = _safe_rate(
            candidate_frame["product_global_reorder_count"],
            candidate_frame["product_global_purchase_count"],
        )

        if history.is_indexed:
            department_counts = category_statistics.loc[
                category_statistics["category_type"] == "department"
            ].set_index("category_id")["purchase_count"]
            aisle_counts = category_statistics.loc[
                category_statistics["category_type"] == "aisle"
            ].set_index("category_id")["purchase_count"]
        else:
            global_catalog_transactions = global_transactions.merge(
                history.catalog()[["product_id", "aisle_id", "department_id"]],
                on="product_id", how="left", validate="many_to_one",
            )
            department_counts = global_catalog_transactions.groupby("department_id").size()
            aisle_counts = global_catalog_transactions.groupby("aisle_id").size()
        customer_catalog_transactions = historical_transactions.merge(
            catalog[["product_id", "aisle_id", "department_id"]],
            on="product_id",
            how="left",
            validate="many_to_one",
        )
        customer_department_counts = customer_catalog_transactions.groupby("department_id").size()
        customer_aisle_counts = customer_catalog_transactions.groupby("aisle_id").size()
        candidate_frame["department_purchase_count"] = candidate_frame["department_id"].map(
            department_counts
        ).fillna(0).astype("int64")
        candidate_frame["aisle_purchase_count"] = candidate_frame["aisle_id"].map(
            aisle_counts
        ).fillna(0).astype("int64")
        denominator = len(historical_transactions)
        candidate_frame["customer_department_affinity"] = (
            candidate_frame["department_id"].map(customer_department_counts).fillna(0) / denominator
            if denominator
            else 0.0
        )
        candidate_frame["customer_aisle_affinity"] = (
            candidate_frame["aisle_id"].map(customer_aisle_counts).fillna(0) / denominator
            if denominator
            else 0.0
        )
        output.append(candidate_frame[FEATURE_COLUMNS])

    result = pd.concat(output, ignore_index=True).sort_values(PREDICTION_KEYS).reset_index(drop=True)
    if result.duplicated(PREDICTION_KEYS).any():
        raise ValueError("feature output contains duplicate prediction rows")
    return result
