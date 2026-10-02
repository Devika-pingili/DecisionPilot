"""Run a bounded, reproducible recommendation demo for one development customer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.features.history import HistoryStore
from backend.recommendation.recommend import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_METADATA_PATH,
    load_model,
    recommend_products,
    write_recommendations,
)


def _development_customer(default_customer_id: int | None) -> int:
    if default_customer_id is not None:
        return int(default_customer_id)
    dataset_metadata_path = REPOSITORY_ROOT / "data" / "processed" / "ml_dev" / "metadata.json"
    metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
    customer_id = metadata.get("customer_ids_first")
    if customer_id is None:
        raise ValueError("The bounded development dataset does not contain a default customer")
    return int(customer_id)


def _prediction_context(history: HistoryStore, customer_id: int) -> tuple[int, int]:
    prior_orders = history.orders.loc[
        (history.orders["user_id"] == customer_id)
        & (history.orders["eval_set"] == "prior")
        & (history.orders["order_number"] > 1)
    ].sort_values(["order_number", "order_id"])
    if prior_orders.empty:
        raise ValueError(f"Customer {customer_id} has no prior-order prediction context")
    target = prior_orders.iloc[-1]
    return int(target["order_id"]), int(target["order_number"])


def _print_table(frame: pd.DataFrame, title: str) -> None:
    print(title)
    print(f"{'Rank':>4}  {'Product':<48} {'Score':>9}  Explanation")
    print("-" * 120)
    for row in frame.itertuples(index=False):
        product = str(row.product_name)
        if len(product) > 48:
            product = product[:45] + "..."
        print(f"{int(row.rank):>4}  {product:<48} {float(row.model_score):>9.6f}  {row.explanation_short}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer-id", type=int, help="Bounded development customer ID")
    parser.add_argument("--target-order-id", type=int, help="Prior order ID defining the prediction cutoff")
    parser.add_argument("--output", type=Path, help="Recommendation CSV output path")
    arguments = parser.parse_args(argv)

    data_dir = REPOSITORY_ROOT / "data"
    processed_dir = data_dir / "processed" / "ml_dev"
    customer_id = _development_customer(arguments.customer_id)
    model = load_model(DEFAULT_MODEL_PATH)
    if not DEFAULT_MODEL_METADATA_PATH.is_file():
        raise FileNotFoundError(f"Model metadata was not found: {DEFAULT_MODEL_METADATA_PATH}")

    with HistoryStore.from_csv(
        data_dir / "raw" / "instacart",
        index_path=data_dir / "processed" / "history.sqlite",
        progress=False,
    ) as history:
        target_order_id, target_order_number = _prediction_context(history, customer_id)
        if arguments.target_order_id is not None:
            target_order_id = arguments.target_order_id
            target = history.target_order(target_order_id)
            if int(target["user_id"]) != customer_id:
                raise ValueError("--target-order-id does not belong to --customer-id")
            target_order_number = int(target["order_number"])
        recommendations = recommend_products(
            history,
            customer_id,
            target_order_id,
            target_order_number,
            model=model,
        )

    output_path = arguments.output or (processed_dir / "recommendations_demo.csv")
    write_recommendations(recommendations.head(10), output_path)
    print(f"Selected customer: {customer_id}")
    print(f"Prediction context order: {target_order_id} (order number {target_order_number})")
    print(f"Candidates scored: {len(recommendations)}")
    _print_table(recommendations.head(5), "Top 5 recommendations")
    _print_table(recommendations.head(10), "Top 10 recommendations")
    print(f"CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())