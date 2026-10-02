# DecisionPilot Data Preparation

DecisionPilot Phase 1 prepares analysis-ready features from the Instacart CSV dataset. This phase does not include a frontend, API, or machine-learning model.

## Setup

Use Python 3.13 or a compatible supported Python 3 release. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the six raw Instacart files under `data/raw/instacart/`:

- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

The preparation pipeline requires `orders.csv`, `order_products__prior.csv`, `products.csv`, and `departments.csv`. The other two files are checked by the inspection script.

## Inspect Raw Data

From the project root:

```powershell
python backend\inspect_data.py
```

The inspection script reads each raw CSV in chunks and prints its row count, columns, inferred types, first five rows, and missing-value counts. It does not modify any files.

## Prepare Data

From the project root:

```powershell
python backend\prepare_data.py
```

The pipeline validates required columns, types, identifiers, value ranges, duplicate keys, and relationships before producing any final output. It writes to a temporary staging directory, validates the generated files, and replaces the existing processed files only after the complete run succeeds. Validation or processing errors return a non-zero exit status.

The pipeline reads `order_products__prior.csv` only. Therefore, the generated purchase, reorder, and time-pattern metrics describe prior orders, not the training order set.

## Processed Files

All outputs are written to `data/processed/`:

- `order_features.csv`: one row per prior order, including customer/order metadata, basket size, reordered item count, reorder rate, and maximum add-to-cart position.
- `product_features.csv`: one row per catalog product, including product metadata, prior purchase count, reorder count, and reorder rate. Products never purchased in prior orders have zero counts and an empty reorder rate.
- `department_features.csv`: one row per department, aggregating purchase count, reorder count, reorder rate, and the number of products purchased at least once.
- `order_time_patterns.csv`: one row per observed day-of-week and hour-of-day combination, with order count, average basket size, and average per-order reorder rate. Instacart does not provide real calendar dates.

## Tests

Run the focused backend tests from the project root:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
