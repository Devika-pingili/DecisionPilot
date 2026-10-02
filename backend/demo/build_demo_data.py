"""Build a bounded customer sample without modifying full local data artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from backend.config import PROJECT_ROOT
from backend.features.history import ORDER_COLUMNS, SOURCE_FILES, TRANSACTION_COLUMNS, HistoryStore
from backend.recommendation.recommend import DEFAULT_MODEL_PATH

SOURCE_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"
DEMO_DATA_ROOT = PROJECT_ROOT / "data" / "demo"
DEMO_CUSTOMER_ID = 1
MINIMUM_PRIOR_ORDERS = 3
TRANSACTION_CHUNK_SIZE = 250_000


def build_demo_data(customer_limit: int = 100) -> Path:
    """Create data/demo from existing Instacart files and the saved model.

    The full source CSVs and full processed database are read-only inputs. The
    generated demo index is built only from the bounded customer subset.
    """
    if customer_limit < 1:
        raise ValueError("customer_limit must be positive")
    if DEMO_DATA_ROOT.exists():
        raise FileExistsError(
            f"Demo output already exists and will not be overwritten: {DEMO_DATA_ROOT}"
        )
    missing_sources = [name for name in SOURCE_FILES if not (SOURCE_RAW_DIR / name).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Missing required source files: {missing_sources}")
    if not DEFAULT_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved model is missing: {DEFAULT_MODEL_PATH}")

    orders = pd.read_csv(SOURCE_RAW_DIR / "orders.csv", usecols=ORDER_COLUMNS)
    prior_orders = orders.loc[orders["eval_set"] == "prior"]
    prior_counts = prior_orders.groupby("user_id").size()
    eligible_ids = sorted(
        int(customer_id)
        for customer_id, count in prior_counts.items()
        if int(count) >= MINIMUM_PRIOR_ORDERS
    )
    if DEMO_CUSTOMER_ID not in eligible_ids:
        raise ValueError("Customer 1 does not have enough prior orders for the demo")

    selected_customer_ids = [DEMO_CUSTOMER_ID]
    selected_customer_ids.extend(
        customer_id
        for customer_id in eligible_ids
        if customer_id != DEMO_CUSTOMER_ID
    )
    selected_customer_ids = selected_customer_ids[:customer_limit]
    selected_orders = orders.loc[orders["user_id"].isin(selected_customer_ids)].copy()
    selected_prior_order_ids = set(
        selected_orders.loc[selected_orders["eval_set"] == "prior", "order_id"].astype(int)
    )

    source_products = pd.read_csv(SOURCE_RAW_DIR / "products.csv")
    aisle_source = pd.read_csv(SOURCE_RAW_DIR / "aisles.csv")
    department_source = pd.read_csv(SOURCE_RAW_DIR / "departments.csv")

    staging_root = Path(tempfile.mkdtemp(prefix=".decisionpilot-demo-", dir=PROJECT_ROOT / "data"))
    raw_dir = staging_root / "raw" / "instacart"
    processed_dir = staging_root / "processed"
    model_dir = processed_dir / "ml_dev"
    raw_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    try:
        selected_orders.to_csv(raw_dir / "orders.csv", index=False)

        transaction_path = raw_dir / "order_products__prior.csv"
        used_product_ids: set[int] = set()
        wrote_transactions = False
        transaction_count = 0
        reader = pd.read_csv(
            SOURCE_RAW_DIR / "order_products__prior.csv",
            usecols=TRANSACTION_COLUMNS,
            chunksize=TRANSACTION_CHUNK_SIZE,
        )
        for chunk in reader:
            selected = chunk.loc[chunk["order_id"].isin(selected_prior_order_ids)]
            if selected.empty:
                continue
            selected.to_csv(
                transaction_path,
                mode="a" if wrote_transactions else "w",
                header=not wrote_transactions,
                index=False,
            )
            wrote_transactions = True
            transaction_count += len(selected)
            used_product_ids.update(selected["product_id"].astype(int).tolist())
        if not wrote_transactions:
            raise ValueError("No prior transactions were found for the selected demo customers")

        products = source_products.loc[
            source_products["product_id"].isin(used_product_ids)
        ].copy()
        if set(products["product_id"].astype(int)) != used_product_ids:
            raise ValueError("Some selected transactions refer to missing catalog products")
        used_aisle_ids = set(products["aisle_id"].astype(int))
        used_department_ids = set(products["department_id"].astype(int))
        aisles = aisle_source.loc[aisle_source["aisle_id"].isin(used_aisle_ids)].copy()
        departments = department_source.loc[
            department_source["department_id"].isin(used_department_ids)
        ].copy()
        if set(aisles["aisle_id"].astype(int)) != used_aisle_ids:
            raise ValueError("Some selected products refer to missing aisles")
        if set(departments["department_id"].astype(int)) != used_department_ids:
            raise ValueError("Some selected products refer to missing departments")

        products.to_csv(raw_dir / "products.csv", index=False)
        aisles.to_csv(raw_dir / "aisles.csv", index=False)
        departments.to_csv(raw_dir / "departments.csv", index=False)

        demo_index_path = processed_dir / "history.sqlite"
        history = HistoryStore.from_csv(raw_dir, index_path=demo_index_path, progress=False)
        try:
            customer_history = history.prior_orders_for_customer(DEMO_CUSTOMER_ID)
            if len(customer_history) < MINIMUM_PRIOR_ORDERS:
                raise ValueError("Demo customer 1 has insufficient indexed prior-order history")
        finally:
            history.close()

        shutil.copy2(DEFAULT_MODEL_PATH, model_dir / "model.joblib")
        os.replace(staging_root, DEMO_DATA_ROOT)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    artifacts = [path for path in DEMO_DATA_ROOT.rglob("*") if path.is_file()]
    print(f"Demo customers: {len(selected_customer_ids)} (includes customer 1)")
    print(f"Prior orders: {int((selected_orders['eval_set'] == 'prior').sum())}")
    print(f"Prior transactions: {transaction_count}")
    for artifact in sorted(artifacts):
        print(f"{artifact.relative_to(DEMO_DATA_ROOT)}: {artifact.stat().st_size:,} bytes")
    print(f"Total data/demo: {sum(path.stat().st_size for path in artifacts):,} bytes")
    return DEMO_DATA_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer-limit", type=int, default=100)
    arguments = parser.parse_args()
    build_demo_data(customer_limit=arguments.customer_limit)


if __name__ == "__main__":
    main()
