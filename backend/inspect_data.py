"""
inspect_data.py  -  DecisionPilot, Step 1: DATA INSPECTION ONLY
 
What this script does:
    For each of the six Instacart CSV files it prints:
        - filename
        - number of rows and columns
        - column names
        - data types
        - first 5 rows
        - missing-value counts
 
What this script does NOT do:
    - It never modifies, moves or deletes the original CSV files (read-only).
    - It never creates fake data.
    - It never loads a whole big file into memory at once. Files are read
      in "chunks" (pieces of 500,000 rows), so even the 32-million-row
      order_products__prior.csv is safe to inspect.
 
How to run (from D:\\DecisionPilot):
    python backend\\inspect_data.py
"""
 
import sys
from pathlib import Path
 
import pandas as pd
 
# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------
 
# Path(__file__) is this script's location (D:\DecisionPilot\backend\inspect_data.py).
# .parent        -> D:\DecisionPilot\backend
# .parent.parent -> D:\DecisionPilot   (the project root)
# Because paths are built from the script's own location, the project still
# works if you move the whole folder to another drive or computer.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"
 
# The six files we expect to find.
CSV_FILES = [
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
]
 
# How many rows to read at a time. Small enough to be safe on memory,
# large enough to be fast.
CHUNK_SIZE = 500_000
 
# Make pandas print ALL columns on one wide screen instead of hiding some with "...".
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
 
 
# ---------------------------------------------------------------------------
# 2. CHECK THAT THE FILES EXIST
# ---------------------------------------------------------------------------
 
def check_files_exist() -> None:
    """Stop with a clear message if the data folder or any CSV is missing."""
    if not DATA_DIR.exists():
        print("ERROR: The data folder was not found:")
        print(f"   {DATA_DIR}")
        print("Check that the Instacart CSV files are inside data\\raw\\instacart\\")
        sys.exit(1)
 
    missing = [name for name in CSV_FILES if not (DATA_DIR / name).is_file()]
    if missing:
        print("ERROR: These files are missing from the data folder:")
        for name in missing:
            print(f"   - {name}")
        print(f"\nExpected location: {DATA_DIR}")
        sys.exit(1)
 
 
# ---------------------------------------------------------------------------
# 3. INSPECT ONE FILE (chunk by chunk)
# ---------------------------------------------------------------------------
 
def inspect_file(filename: str) -> bool:
    """Print the row count, columns, dtypes, first 5 rows and missing values."""
    path = DATA_DIR / filename
 
    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)
 
    total_rows = 0          # running total of rows seen so far
    missing_counts = None   # running total of missing values per column
    first_chunk = None      # we keep only the first chunk (for columns, dtypes, head)
 
    try:
        # chunksize makes pandas return many small DataFrames instead of one huge one.
        # Only ONE chunk is in memory at a time.
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
            if first_chunk is None:
                # head(5) makes a small copy, so the big chunk can be released.
                first_chunk = chunk.head(5).copy()
                column_names = list(chunk.columns)
                dtypes = chunk.dtypes  # data types, as guessed by pandas
 
            total_rows += len(chunk)
 
            # isna().sum() counts missing values in each column for this chunk.
            chunk_missing = chunk.isna().sum()
            if missing_counts is None:
                missing_counts = chunk_missing
            else:
                missing_counts = missing_counts + chunk_missing
 
    except pd.errors.EmptyDataError:
        print("ERROR: This file is empty.")
        return False
    except Exception as error:  # any other reading problem (bad format, permissions...)
        print(f"ERROR: Could not read {filename}: {error}")
        return False
 
    if first_chunk is None:
        print("ERROR: No data found in this file.")
        return False
 
    # --- Print the results -------------------------------------------------
    print(f"Rows:    {total_rows:,}")
    print(f"Columns: {len(column_names)}")
 
    print("\nColumn names:")
    print(column_names)
 
    print("\nData types:")
    print(dtypes.to_string())
 
    print("\nFirst 5 rows:")
    print(first_chunk.to_string(index=False))
 
    print("\nMissing values per column:")
    print(missing_counts.astype(int).to_string())
    return True
 
 
# ---------------------------------------------------------------------------
# 4. MAIN PROGRAM
# ---------------------------------------------------------------------------
 
def main() -> None:
    print("DecisionPilot - Raw Data Inspection")
    print(f"Project folder: {PROJECT_ROOT}")
    print(f"Data folder:    {DATA_DIR}")
 
    check_files_exist()
 
    print("\nNote: order_products__prior.csv has ~32 million rows,")
    print("so it may take a minute or two. This is normal.")
 
    inspection_ok = True
    for filename in CSV_FILES:
        if not inspect_file(filename):
            inspection_ok = False
 
    print("\n" + "=" * 80)
    print("Inspection finished. No files were modified.")
    print("=" * 80)
    if not inspection_ok:
        sys.exit(1)
 
 
# This line makes main() run only when you run this file directly.
if __name__ == "__main__":
    main()
 
