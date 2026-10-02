"""
prepare_data.py  -  DecisionPilot, Phase 2: DATA PREPARATION ONLY
 
What this script does:
    Turns the big raw Instacart CSV files into four smaller, analysis-ready files
    inside data/processed/ :
 
        order_features.csv       one row per (prior) order
        product_features.csv     one row per product
        department_features.csv  one row per department
        order_time_patterns.csv  one row per (day-of-week, hour-of-day)
 
What this script does NOT do:
    - It never modifies the raw CSV files (they are only read).
    - It never invents data: no revenue, price, profit, inventory, discounts,
      customer age/region, and no fake calendar dates. Instacart does not
      contain those things.
    - No machine learning, no LLM, no API, no frontend.
 
Performance idea (important):
    order_products__prior.csv has ~32 million rows. We read it in ONE pass,
    in chunks. During that single pass we build BOTH the order-level totals
    and the product-level totals. Everything else (departments, time
    patterns) is calculated afterwards from those much smaller results,
    so the big file is never scanned a second time.
 
How to run (from D:\\DecisionPilot):
    python backend\\prepare_data.py
"""
 
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional
 
import pandas as pd
 
# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------
 
# This file lives in <project>/backend/, so its parent's parent is the project root.
# Nothing here depends on the folder being called D:\DecisionPilot.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
 
# Number of rows of order_products__prior.csv processed at a time.
# Why chunks? Loading all 32 million rows at once can use several GB of RAM.
# With chunks, only ~1 million rows are in memory at any moment.
CHUNK_SIZE = 1_000_000
 
# Raw files this script needs (aisles.csv and order_products__train.csv are not needed).
REQUIRED_RAW_FILES = [
    "orders.csv",
    "order_products__prior.csv",
    "products.csv",
    "departments.csv",
]
 
# All six raw files: used only to prove at the end that none were changed.
ALL_RAW_FILES = [
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
]
 
# The four files this script generates (existing copies are overwritten).
OUTPUT_FILES = [
    "order_features.csv",
    "product_features.csv",
    "department_features.csv",
    "order_time_patterns.csv",
]

EXPECTED_OUTPUT_COLUMNS = {
    "order_features.csv": [
        "order_id", "user_id", "order_number", "order_dow", "order_hour_of_day",
        "days_since_prior_order", "is_first_order", "basket_size", "reordered_items",
        "reorder_rate", "max_add_to_cart_order",
    ],
    "product_features.csv": [
        "product_id", "product_name", "aisle_id", "department_id", "department",
        "purchase_count", "reorder_count", "reorder_rate",
    ],
    "department_features.csv": [
        "department_id", "department", "purchase_count", "reorder_count",
        "reorder_rate", "unique_products",
    ],
    "order_time_patterns.csv": [
        "order_dow", "order_hour_of_day", "order_count", "average_basket_size",
        "average_reorder_rate",
    ],
}


class DataQualityError(ValueError):
    """Raised when raw data cannot safely produce processed outputs."""
 
 
# ---------------------------------------------------------------------------
# 2. SMALL HELPER FUNCTIONS
# ---------------------------------------------------------------------------
 
def check_raw_files_exist() -> None:
    """Stop with a clear message if a required raw file is missing."""
    missing = [name for name in REQUIRED_RAW_FILES if not (RAW_DIR / name).is_file()]
    if missing:
        print("ERROR: These required raw files were not found:")
        for name in missing:
            print(f"   - {name}")
        print(f"\nExpected location: {RAW_DIR}")
        sys.exit(1)
 
 
def snapshot_raw_files() -> dict:
    """Record size and last-modified time of each raw file.
    We compare this before/after to prove the raw data was not touched."""
    snapshot = {}
    for name in ALL_RAW_FILES:
        path = RAW_DIR / name
        if path.is_file():
            info = path.stat()
            snapshot[name] = (info.st_size, info.st_mtime_ns)
    return snapshot
 
 
def save_csv(df: pd.DataFrame, filename: str, output_dir: Optional[Path] = None) -> None:
    """Save a generated CSV into the supplied output directory."""
    path = (output_dir or PROCESSED_DIR) / filename
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        print(f"ERROR: Cannot write {filename}. Is it open in Excel or another program?")
        print("Close it and run the script again.")
        sys.exit(1)
    print(f"  saved {filename}  ({len(df):,} rows)")


def require_columns(frame: pd.DataFrame, required: list[str], filename: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataQualityError(
            f"{filename} is missing required columns: {', '.join(missing)}"
        )


def require_non_empty(frame: pd.DataFrame, filename: str) -> None:
    if frame.empty:
        raise DataQualityError(f"{filename} is empty")


def require_positive_ids(frame: pd.DataFrame, columns: list[str], filename: str) -> None:
    for column in columns:
        invalid = frame[column].isna() | (frame[column] <= 0)
        if invalid.any():
            raise DataQualityError(
                f"{filename}.{column} contains {int(invalid.sum()):,} missing or non-positive IDs"
            )


def require_integer_columns(frame: pd.DataFrame, columns: list[str], filename: str) -> None:
    for column in columns:
        if not pd.api.types.is_integer_dtype(frame[column]):
            raise DataQualityError(f"{filename}.{column} must contain integer values")


def validate_orders(orders: pd.DataFrame) -> None:
    filename = "orders.csv"
    require_non_empty(orders, filename)
    require_columns(orders, [
        "order_id", "user_id", "order_number", "order_dow", "order_hour_of_day",
        "days_since_prior_order", "eval_set",
    ], filename)
    require_integer_columns(
        orders,
        ["order_id", "user_id", "order_number", "order_dow", "order_hour_of_day"],
        filename,
    )
    require_positive_ids(orders, ["order_id", "user_id"], filename)
    if orders["order_id"].duplicated().any():
        raise DataQualityError("orders.csv contains duplicate order_id values")
    if (orders["order_number"] < 1).any():
        raise DataQualityError("orders.csv.order_number must be at least 1")
    if not orders["order_dow"].between(0, 6).all():
        raise DataQualityError("orders.csv.order_dow must be between 0 and 6")
    if not orders["order_hour_of_day"].between(0, 23).all():
        raise DataQualityError("orders.csv.order_hour_of_day must be between 0 and 23")
    if orders["days_since_prior_order"].notna().any() and (
        orders["days_since_prior_order"].dropna() < 0
    ).any():
        raise DataQualityError("orders.csv.days_since_prior_order cannot be negative")
    if orders["eval_set"].isna().any() or not orders["eval_set"].isin(
        ["prior", "train", "test"]
    ).all():
        raise DataQualityError("orders.csv.eval_set contains missing or unexpected values")


def load_catalogs() -> tuple[pd.DataFrame, pd.DataFrame]:
    products_path = RAW_DIR / "products.csv"
    departments_path = RAW_DIR / "departments.csv"
    try:
        products = pd.read_csv(products_path)
        departments = pd.read_csv(departments_path)
    except pd.errors.EmptyDataError as error:
        raise DataQualityError(f"Empty catalog file: {error}") from error
    except (pd.errors.ParserError, ValueError) as error:
        raise DataQualityError(f"Could not read catalog file: {error}") from error

    require_non_empty(products, "products.csv")
    require_non_empty(departments, "departments.csv")
    require_columns(products, ["product_id", "product_name", "aisle_id", "department_id"], "products.csv")
    require_columns(departments, ["department_id", "department"], "departments.csv")
    if products["product_name"].isna().any() or products["product_name"].astype(str).str.strip().eq("").any():
        raise DataQualityError("products.csv.product_name contains missing or blank values")
    if departments["department"].isna().any() or departments["department"].astype(str).str.strip().eq("").any():
        raise DataQualityError("departments.csv.department contains missing or blank values")
    require_integer_columns(products, ["product_id", "aisle_id", "department_id"], "products.csv")
    require_integer_columns(departments, ["department_id"], "departments.csv")
    require_positive_ids(products, ["product_id", "aisle_id", "department_id"], "products.csv")
    require_positive_ids(departments, ["department_id"], "departments.csv")
    if products["product_id"].duplicated().any():
        raise DataQualityError("products.csv contains duplicate product_id values")
    if departments["department_id"].duplicated().any():
        raise DataQualityError("departments.csv contains duplicate department_id values")
    if not products["department_id"].isin(departments["department_id"]).all():
        raise DataQualityError("products.csv contains department_id values missing from departments.csv")
    return products, departments
 
 
# ---------------------------------------------------------------------------
# 3. LOAD ORDERS
# ---------------------------------------------------------------------------
 
def load_orders() -> pd.DataFrame:
    """Read orders.csv with ONLY the columns we need.
    Small integer types (int8, int16, int32) keep memory use low for 3.4M rows."""
    columns = [
        "order_id", "user_id", "order_number", "order_dow",
        "order_hour_of_day", "days_since_prior_order", "eval_set",
    ]
    dtypes = {
        "order_id": "int32",
        "user_id": "int32",
        "order_number": "int16",
        "order_dow": "int8",
        "order_hour_of_day": "int8",
        "days_since_prior_order": "float32",  # float because it contains missing values (NaN)
        "eval_set": "category",
    }
    try:
        orders = pd.read_csv(RAW_DIR / "orders.csv", usecols=columns, dtype=dtypes)
    except pd.errors.EmptyDataError as error:
        raise DataQualityError("orders.csv is empty") from error
    except ValueError as error:
        raise DataQualityError(f"orders.csv has missing columns or invalid values: {error}") from error
    except pd.errors.ParserError as error:
        raise DataQualityError(f"Could not parse orders.csv: {error}") from error
    validate_orders(orders)
    print(f"  loaded orders.csv: {len(orders):,} rows")
    return orders
 
 
# ---------------------------------------------------------------------------
# 4. THE ONE AND ONLY PASS THROUGH order_products__prior.csv
# ---------------------------------------------------------------------------
 
def scan_prior_transactions():
    """Read order_products__prior.csv ONCE, chunk by chunk.
 
    For every chunk we calculate two small summaries:
        - per order:   items in order, reordered items, max add_to_cart_order
        - per product: times purchased, times reordered
    and combine them as we go.
 
    Returns:
        order_totals   (DataFrame, one row per order)
        product_totals (DataFrame, one row per product)
    """
    path = RAW_DIR / "order_products__prior.csv"
    dtypes = {
        "order_id": "int32",
        "product_id": "int32",
        "add_to_cart_order": "int32",
        "reordered": "int32",
    }
 
    order_parts = []       # list of small per-chunk order summaries
    product_total = None   # running product totals (only ~50,000 rows, tiny)
    rows_done = 0
 
    try:
        reader = pd.read_csv(path, dtype=dtypes, chunksize=CHUNK_SIZE)
    except pd.errors.EmptyDataError as error:
        raise DataQualityError("order_products__prior.csv is empty") from error
    except ValueError as error:
        raise DataQualityError(
            f"order_products__prior.csv has missing columns or invalid values: {error}"
        ) from error
    except pd.errors.ParserError as error:
        raise DataQualityError(f"Could not parse order_products__prior.csv: {error}") from error

    for chunk_number, chunk in enumerate(reader, start=1):
        require_columns(
            chunk, ["order_id", "product_id", "add_to_cart_order", "reordered"],
            "order_products__prior.csv",
        )
        require_non_empty(chunk, "order_products__prior.csv")
        require_integer_columns(
            chunk, ["order_id", "product_id", "add_to_cart_order", "reordered"],
            "order_products__prior.csv",
        )
        require_positive_ids(chunk, ["order_id", "product_id", "add_to_cart_order"], "order_products__prior.csv")
        if not chunk["reordered"].isin([0, 1]).all():
            raise DataQualityError("order_products__prior.csv.reordered must contain only 0 or 1")
        if chunk.duplicated(["order_id", "product_id"]).any():
            raise DataQualityError(
                "order_products__prior.csv contains duplicate (order_id, product_id) pairs"
            )
        # --- order-level summary for this chunk ---
        by_order = chunk.groupby("order_id").agg(
            basket_size=("product_id", "size"),                   # items in the order
            reordered_items=("reordered", "sum"),                 # how many were reorders
            max_add_to_cart_order=("add_to_cart_order", "max"),
        )
        order_parts.append(by_order)
 
        # --- product-level summary for this chunk ---
        by_product = chunk.groupby("product_id").agg(
            purchase_count=("order_id", "size"),
            reorder_count=("reordered", "sum"),
        )
        if product_total is None:
            product_total = by_product
        else:
            # add() lines up rows by product_id; fill_value=0 handles products
            # that appear in one chunk but not the other.
            product_total = product_total.add(by_product, fill_value=0)
 
        rows_done += len(chunk)
        print(f"  chunk {chunk_number}: {rows_done:,} rows processed")
 
    # An order's rows can be split across two chunks (when a chunk boundary falls
    # in the middle of an order). So we merge the partial summaries per order:
    # add up counts, take the maximum of the max.
    if not order_parts:
        raise DataQualityError("order_products__prior.csv is empty")

    all_orders = pd.concat(order_parts)
    order_totals = all_orders.groupby(level=0).agg(
        basket_size=("basket_size", "sum"),
        reordered_items=("reordered_items", "sum"),
        max_add_to_cart_order=("max_add_to_cart_order", "max"),
    ).reset_index()
 
    product_totals = product_total.astype("int64").reset_index()
    return order_totals, product_totals
 
 
# ---------------------------------------------------------------------------
# 5. ORDER FEATURES
# ---------------------------------------------------------------------------
 
def process_order_features(
    orders: pd.DataFrame,
    order_totals: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Join the per-order totals with order information from orders.csv."""
    # order_products__prior.csv only covers orders whose eval_set is "prior".
    prior_orders = orders.loc[
        orders["eval_set"] == "prior",
        ["order_id", "user_id", "order_number", "order_dow",
         "order_hour_of_day", "days_since_prior_order"],
    ]
 
    features = order_totals.merge(prior_orders, on="order_id", how="inner")
    if len(features) != len(order_totals):
        missing_count = len(order_totals) - len(features)
        raise DataQualityError(
            f"{missing_count:,} prior transaction orders were not found in orders.csv"
        )
 
    # reorder_rate = share of items in the order that were reorders (0 to 1).
    features["reorder_rate"] = (features["reordered_items"] / features["basket_size"]).round(4)
 
    # days_since_prior_order is naturally empty for a user's FIRST order
    # (there is no earlier order). That is expected, NOT an error, and we keep
    # it empty. is_first_order makes this explicit.
    features["is_first_order"] = features["order_number"] == 1
 
    # Sanity check: are the missing values exactly the first orders?
    missing_days = features["days_since_prior_order"].isna()
    if (missing_days == features["is_first_order"]).all():
        print(f"  days_since_prior_order is empty for {int(missing_days.sum()):,} orders "
              "- exactly the first orders (expected, left as-is)")
    else:
        print("  NOTE: some empty days_since_prior_order values are NOT first orders - please check")
 
    features = features[[
        "order_id", "user_id", "order_number", "order_dow", "order_hour_of_day",
        "days_since_prior_order", "is_first_order",
        "basket_size", "reordered_items", "reorder_rate", "max_add_to_cart_order",
    ]]
    save_csv(features, "order_features.csv", output_dir)
    return features
 
 
# ---------------------------------------------------------------------------
# 6. PRODUCT FEATURES
# ---------------------------------------------------------------------------
 
def process_product_features(
    product_totals: pd.DataFrame,
    products: Optional[pd.DataFrame] = None,
    departments: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Add product names, aisle ids and department names to the product totals."""
    if products is None or departments is None:
        products, departments = load_catalogs()

    unknown_products = set(product_totals["product_id"]) - set(products["product_id"])
    if unknown_products:
        raise DataQualityError(
            f"order_products__prior.csv contains {len(unknown_products):,} product_id values "
            "missing from products.csv"
        )
 
    # Start from the product catalog so every product appears, even if it was never bought.
    features = products.merge(product_totals, on="product_id", how="left")
    features[["purchase_count", "reorder_count"]] = (
        features[["purchase_count", "reorder_count"]].fillna(0).astype("int64")
    )
    features = features.merge(departments, on="department_id", how="left")
    if features["department"].isna().any():
        raise DataQualityError("products.csv contains department_id values missing from departments.csv")
 
    # reorder_rate = reorder_count / purchase_count.
    # A never-purchased product has 0/0, which stays empty (NaN) instead of a fake 0.
    features["reorder_rate"] = (features["reorder_count"] / features["purchase_count"]).round(4)
 
    features = features[[
        "product_id", "product_name", "aisle_id", "department_id", "department",
        "purchase_count", "reorder_count", "reorder_rate",
    ]].sort_values(["purchase_count", "product_id"], ascending=[False, True])
 
    save_csv(features, "product_features.csv", output_dir)
    return features
 
 
# ---------------------------------------------------------------------------
# 7. DEPARTMENT FEATURES
# ---------------------------------------------------------------------------
 
def process_department_features(
    product_features: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Roll the product totals up to department level.
    No extra pass over the big file is needed: a department's purchases are just
    the sum of its products' purchases."""
    data = product_features.copy()
    # unique_products = number of products in this department that were
    # actually purchased at least once in the prior orders.
    data["is_purchased"] = (data["purchase_count"] > 0).astype(int)
 
    departments = data.groupby(["department_id", "department"], as_index=False).agg(
        purchase_count=("purchase_count", "sum"),
        reorder_count=("reorder_count", "sum"),
        unique_products=("is_purchased", "sum"),
    )
    departments["reorder_rate"] = (
        departments["reorder_count"] / departments["purchase_count"]
    ).round(4)
 
    departments = departments[[
        "department_id", "department", "purchase_count",
        "reorder_count", "reorder_rate", "unique_products",
    ]].sort_values("purchase_count", ascending=False)
 
    save_csv(departments, "department_features.csv", output_dir)
    return departments
 
 
# ---------------------------------------------------------------------------
# 8. TIME PATTERNS (day of week x hour of day)
# ---------------------------------------------------------------------------
 
def process_time_patterns(
    order_features: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Summarise orders by day of week and hour of day.
    Instacart has NO real calendar dates, so these two columns are the only
    time information we can honestly use. (The dataset does not document which
    number is which weekday, so we keep the raw 0-6 values.)"""
    patterns = order_features.groupby(["order_dow", "order_hour_of_day"], as_index=False).agg(
        order_count=("order_id", "size"),
        average_basket_size=("basket_size", "mean"),
        average_reorder_rate=("reorder_rate", "mean"),  # average of the per-order reorder rates
    )
    patterns["average_basket_size"] = patterns["average_basket_size"].round(4)
    patterns["average_reorder_rate"] = patterns["average_reorder_rate"].round(4)
    patterns = patterns.sort_values(["order_dow", "order_hour_of_day"])
 
    save_csv(patterns, "order_time_patterns.csv", output_dir)
    return patterns
 
 
# ---------------------------------------------------------------------------
# 9. VALIDATION
# ---------------------------------------------------------------------------
 
def summarize_csv(path: Path):
    """Read a finished CSV back in chunks (memory-safe) and return
    (row count, column names, missing values per column)."""
    rows = 0
    columns = None
    missing = None
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        rows += len(chunk)
        columns = list(chunk.columns)
        chunk_missing = chunk.isna().sum()
        missing = chunk_missing if missing is None else missing + chunk_missing
    return rows, columns, missing
 
 
def validate_outputs(raw_before: dict, output_dir: Optional[Path] = None) -> bool:
    """Check that every output file exists, then print rows, columns and missing values."""
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
 
    all_ok = True
    for name in OUTPUT_FILES:
        path = (output_dir or PROCESSED_DIR) / name
        if not path.is_file():
            print(f"\n✗ {name} was NOT created")
            all_ok = False
            continue
 
        rows, columns, missing = summarize_csv(path)
        print(f"\n✓ {name} exists")
        print(f"  rows:    {rows:,}")
        print(f"  columns: {columns}")

        if rows == 0:
            print("  ERROR: output is empty")
            all_ok = False
        if columns != EXPECTED_OUTPUT_COLUMNS[name]:
            print(f"  ERROR: unexpected columns; expected {EXPECTED_OUTPUT_COLUMNS[name]}")
            all_ok = False
 
        missing = missing[missing > 0]
        if missing.empty:
            print("  missing values: none")
        else:
            print("  missing values:")
            for column, count in missing.items():
                print(f"    {column}: {int(count):,}")
 
    print("\nNotes on expected missing values:")
    print("  - days_since_prior_order is empty for first orders (naturally no earlier order).")
    print("  - reorder_rate in product_features is empty only for never-purchased products.")
 
    # Prove the raw files were not modified: size and modified-time must be unchanged.
    print()
    if snapshot_raw_files() == raw_before:
        print("✓ Raw files were not modified (sizes and modified-times unchanged)")
    else:
        print("✗ WARNING: a raw file changed while the script ran - please check!")
        all_ok = False
 
    return all_ok


def commit_outputs(staging_dir: Path) -> None:
    """Replace all outputs together, restoring old files if a move fails."""
    backup_dir = Path(tempfile.mkdtemp(prefix=".prepare-backup-", dir=PROCESSED_DIR.parent))
    moved_new: list[Path] = []
    moved_old: list[tuple[Path, Path]] = []
    try:
        for name in OUTPUT_FILES:
            destination = PROCESSED_DIR / name
            if destination.exists():
                backup = backup_dir / name
                os.replace(destination, backup)
                moved_old.append((destination, backup))
        for name in OUTPUT_FILES:
            destination = PROCESSED_DIR / name
            os.replace(staging_dir / name, destination)
            moved_new.append(destination)
    except OSError as error:
        for destination in moved_new:
            if destination.exists():
                destination.unlink()
        for destination, backup in moved_old:
            if backup.exists():
                os.replace(backup, destination)
        raise DataQualityError(f"Could not commit processed outputs: {error}") from error
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)
        shutil.rmtree(staging_dir, ignore_errors=True)
 
 
# ---------------------------------------------------------------------------
# 10. MAIN PROGRAM
# ---------------------------------------------------------------------------
 
def main() -> None:
    # Lets the ✓ symbols print correctly in more Windows terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
 
    start_time = time.perf_counter()
    print("DecisionPilot - Data Preparation (Phase 2)")
    print(f"Project folder: {PROJECT_ROOT}")
    print(f"Raw data:       {RAW_DIR}")
    print(f"Output folder:  {PROCESSED_DIR}")
 
    staging_dir = None
    try:
        check_raw_files_exist()
        raw_before = snapshot_raw_files()

        # Build all outputs outside the final directory until validation succeeds.
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=".prepare-", dir=PROCESSED_DIR.parent))

        print("\n[1/3] Loading and validating input catalogs ...")
        orders = load_orders()
        products, departments = load_catalogs()

        print("\n[2/3] Reading order_products__prior.csv ONCE in chunks (may take a few minutes) ...")
        order_totals, product_totals = scan_prior_transactions()

        prior_order_ids = set(orders.loc[orders["eval_set"] == "prior", "order_id"])
        missing_order_ids = ~order_totals["order_id"].isin(prior_order_ids)
        if missing_order_ids.any():
            raise DataQualityError(
                f"{int(missing_order_ids.sum()):,} transaction orders are not marked as prior in orders.csv"
            )

        print("\n[3/3] Building and validating processed files ...")
        order_features = process_order_features(orders, order_totals, staging_dir)
        product_features = process_product_features(
            product_totals, products, departments, staging_dir
        )
        process_department_features(product_features, staging_dir)
        process_time_patterns(order_features, staging_dir)

        if not validate_outputs(raw_before, staging_dir):
            raise DataQualityError("Generated outputs failed validation; existing outputs were preserved")

        commit_outputs(staging_dir)
        staging_dir = None

        elapsed = time.perf_counter() - start_time
        print("\n" + "=" * 70)
        print("DecisionPilot - Data Preparation Complete")
        print("\nProcessed files:")
        for name in OUTPUT_FILES:
            print(f"✓ {name}")
        print("\nRaw data was not modified.")
        print(f"\nTime taken: {elapsed:.0f} seconds")
        print("=" * 70)
    except DataQualityError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"\nERROR: Unexpected pipeline failure: {error}", file=sys.stderr)
        sys.exit(1)
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
 
 
if __name__ == "__main__":
    main()
 
