"""Sparse, indexed point-in-time access to Instacart prior-order history."""

import os
import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


ORDER_COLUMNS = [
    "order_id", "user_id", "order_number", "order_dow",
    "order_hour_of_day", "days_since_prior_order", "eval_set",
]
TRANSACTION_COLUMNS = ["order_id", "product_id", "add_to_cart_order", "reordered"]
INDEX_FORMAT_VERSION = 1
PREPARATION_SCHEMA_VERSION = 1
INDEX_TABLES = {
    "metadata",
    "orders",
    "prior_transactions",
    "global_product_stats",
    "global_category_stats",
}
SOURCE_FILES = ["orders.csv", "order_products__prior.csv", "products.csv", "aisles.csv", "departments.csv"]
PREPARATION_CONFIG = {
    "cutoff_rule": "order_number < target_order_number",
    "transaction_source": "order_products__prior.csv",
    "transaction_columns": TRANSACTION_COLUMNS,
    "aggregate_tables": ["global_product_stats", "global_category_stats"],
}


class HistoryDataError(ValueError):
    """Raised when history inputs cannot support point-in-time features."""


class IndexCompatibilityError(HistoryDataError):
    """Raised when a persisted history index cannot safely be used."""


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise HistoryDataError(f"{name} is missing required columns: {', '.join(missing)}")


def _require_non_empty(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise HistoryDataError(f"{name} is empty")


def _placeholders(count: int) -> str:
    return ",".join("?" for _ in range(count))


def _source_metadata(raw_dir: Path) -> dict[str, str]:
    metadata = {"source_files": json.dumps(SOURCE_FILES, separators=(",", ":"))}
    for filename in SOURCE_FILES:
        path = Path(raw_dir) / filename
        if not path.is_file():
            raise IndexCompatibilityError(f"Required source file is missing: {path}")
        stat = path.stat()
        metadata[f"source.{filename}"] = json.dumps(
            {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            sort_keys=True,
            separators=(",", ":"),
        )
    return metadata


class HistoryIndex:
    """Local SQLite representation built once from chunked prior transactions."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        try:
            self.validate()
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "HistoryIndex":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _read_sql(self, query: str, params: Sequence[object] = ()) -> pd.DataFrame:
        return pd.read_sql_query(query, self.connection, params=params)

    def validate(
        self,
        raw_dir: Optional[Path] = None,
        expected_config: Optional[dict[str, object]] = None,
    ) -> bool:
        """Validate index structure, format, configuration, and optional sources."""
        tables = {
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing_tables = sorted(INDEX_TABLES - tables)
        if missing_tables:
            raise IndexCompatibilityError(
                f"History index is missing required tables: {', '.join(missing_tables)}"
            )
        metadata = {
            row["key"]: row["value"]
            for row in self.connection.execute("SELECT key, value FROM metadata")
        }
        required_metadata = {
            "index_format_version",
            "preparation_schema_version",
            "preparation_config",
            "source_files",
            *[f"source.{filename}" for filename in SOURCE_FILES],
        }
        missing_metadata = sorted(required_metadata - set(metadata))
        if missing_metadata:
            raise IndexCompatibilityError(
                "History index is missing required metadata: " + ", ".join(missing_metadata)
            )
        if metadata["index_format_version"] != str(INDEX_FORMAT_VERSION):
            raise IndexCompatibilityError(
                "History index format version is incompatible: "
                f"{metadata['index_format_version']} != {INDEX_FORMAT_VERSION}"
            )
        if metadata["preparation_schema_version"] != str(PREPARATION_SCHEMA_VERSION):
            raise IndexCompatibilityError(
                "History preparation schema version is incompatible: "
                f"{metadata['preparation_schema_version']} != {PREPARATION_SCHEMA_VERSION}"
            )
        try:
            stored_config = json.loads(metadata["preparation_config"])
            stored_source_files = json.loads(metadata["source_files"])
        except json.JSONDecodeError as error:
            raise IndexCompatibilityError("History index metadata contains invalid JSON") from error
        current_config = expected_config or PREPARATION_CONFIG
        if stored_config != current_config:
            raise IndexCompatibilityError(
                "History index preparation configuration is incompatible"
            )
        if stored_source_files != SOURCE_FILES:
            raise IndexCompatibilityError("History index source-file set is incompatible")
        if raw_dir is not None:
            current_sources = _source_metadata(Path(raw_dir))
            for key, value in current_sources.items():
                if metadata.get(key) != value:
                    raise IndexCompatibilityError(
                        f"History index source metadata mismatch for {key}"
                    )
        return True

    def orders_before(self, customer_id: int, target_order_number: int) -> pd.DataFrame:
        return self._read_sql(
            """
            SELECT order_id, user_id, order_number, order_dow,
                   order_hour_of_day, days_since_prior_order, eval_set
            FROM orders
            WHERE user_id = ? AND eval_set = 'prior' AND order_number < ?
            ORDER BY order_number, order_id
            """,
            (customer_id, target_order_number),
        )

    def transactions_for_customer_before(
        self, customer_id: int, target_order_number: int
    ) -> pd.DataFrame:
        return self._read_sql(
            """
            SELECT t.order_id, t.product_id, t.add_to_cart_order, t.reordered,
                   o.user_id, o.order_number, o.days_since_prior_order
            FROM prior_transactions AS t
            JOIN orders AS o ON o.order_id = t.order_id
            WHERE o.user_id = ? AND o.eval_set = 'prior'
              AND o.order_number < ?
            ORDER BY o.order_number, t.order_id, t.product_id
            """,
            (customer_id, target_order_number),
        )

    def transactions_for_order(self, order_id: int) -> pd.DataFrame:
        return self._read_sql(
            """
            SELECT order_id, product_id, add_to_cart_order, reordered
            FROM prior_transactions WHERE order_id = ? ORDER BY product_id
            """,
            (order_id,),
        )

    def product_statistics(
        self, target_order_number: int, product_ids: Optional[Sequence[int]] = None
    ) -> pd.DataFrame:
        query = """
            SELECT product_id, aisle_id, department_id,
                   purchase_count AS product_global_purchase_count,
                   reorder_count AS product_global_reorder_count
            FROM global_product_stats
            WHERE cutoff_order_number = ?
        """
        params: list[object] = [target_order_number]
        if product_ids is not None:
            ids = sorted(set(int(product_id) for product_id in product_ids))
            if not ids:
                return pd.DataFrame(columns=[
                    "product_id", "aisle_id", "department_id",
                    "product_global_purchase_count", "product_global_reorder_count",
                ])
            query += f" AND product_id IN ({_placeholders(len(ids))})"
            params.extend(ids)
        query += " ORDER BY product_id"
        return self._read_sql(query, params)

    def top_products(
        self,
        target_order_number: int,
        limit: int,
        departments: Optional[set[int]] = None,
        aisles: Optional[set[int]] = None,
    ) -> pd.DataFrame:
        query = """
            SELECT product_id, aisle_id, department_id,
                   purchase_count AS score
            FROM global_product_stats
            WHERE cutoff_order_number = ?
        """
        params: list[object] = [target_order_number]
        filters: list[str] = []
        if departments:
            values = sorted(int(value) for value in departments)
            filters.append(f"department_id IN ({_placeholders(len(values))})")
            params.extend(values)
        if aisles:
            values = sorted(int(value) for value in aisles)
            filters.append(f"aisle_id IN ({_placeholders(len(values))})")
            params.extend(values)
        if filters:
            query += " AND (" + " OR ".join(filters) + ")"
        query += " ORDER BY score DESC, product_id LIMIT ?"
        params.append(limit)
        return self._read_sql(query, params)

    def category_statistics(self, target_order_number: int) -> pd.DataFrame:
        return self._read_sql(
            """
            SELECT category_type, category_id, purchase_count
            FROM global_category_stats
            WHERE cutoff_order_number = ?
            ORDER BY category_type, category_id
            """,
            (target_order_number,),
        )


def prepare_history_index(
    raw_dir: Path,
    output_path: Path,
    *,
    chunk_size: int = 1_000_000,
    progress: bool = True,
) -> Path:
    """Build a deterministic local index from prior transactions in chunks."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-", suffix=".sqlite", dir=output_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        orders = pd.read_csv(raw_dir / "orders.csv", usecols=ORDER_COLUMNS)
        products = pd.read_csv(raw_dir / "products.csv")
        _require_non_empty(orders, "orders")
        _require_non_empty(products, "products")
        _require_columns(orders, ORDER_COLUMNS, "orders")
        _require_columns(products, ["product_id", "product_name", "aisle_id", "department_id"], "products")
        prior_orders = orders.loc[orders["eval_set"] == "prior", ORDER_COLUMNS].copy()
        order_lookup = prior_orders.set_index("order_id")
        product_lookup = products.set_index("product_id")[["aisle_id", "department_id"]]
        max_order_number = int(orders["order_number"].max())
        metadata = {
            "index_format_version": str(INDEX_FORMAT_VERSION),
            "preparation_schema_version": str(PREPARATION_SCHEMA_VERSION),
            "preparation_config": json.dumps(
                PREPARATION_CONFIG, sort_keys=True, separators=(",", ":")
            ),
            **_source_metadata(raw_dir),
        }

        connection = sqlite3.connect(temporary_path)
        connection.executescript(
            """
            PRAGMA journal_mode = DELETE;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
                order_number INTEGER NOT NULL, order_dow INTEGER NOT NULL,
                order_hour_of_day INTEGER NOT NULL, days_since_prior_order REAL,
                eval_set TEXT NOT NULL
            );
            CREATE TABLE prior_transactions (
                order_id INTEGER NOT NULL, product_id INTEGER NOT NULL,
                add_to_cart_order INTEGER NOT NULL, reordered INTEGER NOT NULL,
                PRIMARY KEY (order_id, product_id)
            );
            CREATE INDEX idx_orders_customer_number ON orders(user_id, order_number);
            CREATE INDEX idx_transactions_order ON prior_transactions(order_id);
            CREATE TABLE product_events (
                order_number INTEGER NOT NULL, product_id INTEGER NOT NULL,
                aisle_id INTEGER NOT NULL, department_id INTEGER NOT NULL,
                purchase_count INTEGER NOT NULL, reorder_count INTEGER NOT NULL,
                PRIMARY KEY (order_number, product_id)
            );
            CREATE TABLE category_events (
                order_number INTEGER NOT NULL, category_type TEXT NOT NULL,
                category_id INTEGER NOT NULL, purchase_count INTEGER NOT NULL,
                PRIMARY KEY (order_number, category_type, category_id)
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)",
            prior_orders.itertuples(index=False, name=None),
        )
        connection.commit()

        reader = pd.read_csv(
            raw_dir / "order_products__prior.csv",
            usecols=TRANSACTION_COLUMNS,
            chunksize=chunk_size,
        )
        rows_done = 0
        for chunk_number, chunk in enumerate(reader, start=1):
            _require_columns(chunk, TRANSACTION_COLUMNS, "order_products__prior.csv")
            if chunk.empty:
                raise HistoryDataError("order_products__prior.csv contains an empty chunk")
            if chunk.duplicated(["order_id", "product_id"]).any():
                raise HistoryDataError("prior transactions contain duplicate order/product rows")
            if not chunk["reordered"].isin([0, 1]).all():
                raise HistoryDataError("prior_transactions.reordered must contain only 0 or 1")
            if not set(chunk["order_id"]).issubset(set(order_lookup.index)):
                raise HistoryDataError("prior transactions contain non-prior order IDs")
            if not set(chunk["product_id"]).issubset(set(product_lookup.index)):
                raise HistoryDataError("prior transactions contain unknown product IDs")

            chunk_with_order = chunk.join(order_lookup[["order_number"]], on="order_id", how="inner")
            chunk_with_product = chunk_with_order.join(product_lookup, on="product_id", how="inner")
            connection.executemany(
                "INSERT INTO prior_transactions VALUES (?, ?, ?, ?)",
                chunk[TRANSACTION_COLUMNS].itertuples(index=False, name=None),
            )
            product_events = chunk_with_product.groupby(
                ["order_number", "product_id", "aisle_id", "department_id"], as_index=False
            ).agg(purchase_count=("order_id", "size"), reorder_count=("reordered", "sum"))
            connection.executemany(
                """
                INSERT INTO product_events VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_number, product_id) DO UPDATE SET
                    purchase_count = purchase_count + excluded.purchase_count,
                    reorder_count = reorder_count + excluded.reorder_count
                """,
                product_events.itertuples(index=False, name=None),
            )
            department_events = chunk_with_product.groupby(
                ["order_number", "department_id"], as_index=False
            ).size().rename(columns={"size": "purchase_count"})
            aisle_events = chunk_with_product.groupby(
                ["order_number", "aisle_id"], as_index=False
            ).size().rename(columns={"size": "purchase_count"})
            category_rows = [
                (int(row.order_number), "department", int(row.department_id), int(row.purchase_count))
                for row in department_events.itertuples(index=False)
            ] + [
                (int(row.order_number), "aisle", int(row.aisle_id), int(row.purchase_count))
                for row in aisle_events.itertuples(index=False)
            ]
            connection.executemany(
                """
                INSERT INTO category_events VALUES (?, ?, ?, ?)
                ON CONFLICT(order_number, category_type, category_id) DO UPDATE SET
                    purchase_count = purchase_count + excluded.purchase_count
                """,
                category_rows,
            )
            rows_done += len(chunk)
            connection.commit()
            if progress:
                print(f"  history chunk {chunk_number}: {rows_done:,} rows indexed")

        connection.executescript(
            """
            CREATE TABLE global_product_stats (
                cutoff_order_number INTEGER NOT NULL, product_id INTEGER NOT NULL,
                aisle_id INTEGER NOT NULL, department_id INTEGER NOT NULL,
                purchase_count INTEGER NOT NULL, reorder_count INTEGER NOT NULL,
                PRIMARY KEY (cutoff_order_number, product_id)
            );
            CREATE INDEX idx_global_product_lookup
                ON global_product_stats(cutoff_order_number, product_id);
            CREATE TABLE global_category_stats (
                cutoff_order_number INTEGER NOT NULL, category_type TEXT NOT NULL,
                category_id INTEGER NOT NULL, purchase_count INTEGER NOT NULL,
                PRIMARY KEY (cutoff_order_number, category_type, category_id)
            );
            """
        )
        product_totals = pd.DataFrame(columns=[
            "product_id", "aisle_id", "department_id", "purchase_count", "reorder_count"
        ]).set_index("product_id")
        category_totals = pd.DataFrame(columns=["category_type", "category_id", "purchase_count"])
        for cutoff in range(1, max_order_number + 2):
            product_rows = pd.read_sql_query(
                "SELECT product_id, aisle_id, department_id, purchase_count, reorder_count "
                "FROM product_events WHERE order_number = ?",
                connection, params=(cutoff - 1,),
            ).set_index("product_id")
            if not product_rows.empty:
                product_totals = product_rows if product_totals.empty else product_totals.add(product_rows, fill_value=0)
            if not product_totals.empty:
                connection.executemany(
                    "INSERT INTO global_product_stats VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (cutoff, int(product_id), int(row.aisle_id), int(row.department_id),
                         int(row.purchase_count), int(row.reorder_count))
                        for product_id, row in product_totals.iterrows()
                    ],
                )

            category_rows = pd.read_sql_query(
                "SELECT category_type, category_id, purchase_count FROM category_events "
                "WHERE order_number = ?",
                connection, params=(cutoff - 1,),
            )
            if not category_rows.empty:
                category_totals = pd.concat([category_totals, category_rows], ignore_index=True)
                category_totals = category_totals.groupby(
                    ["category_type", "category_id"], as_index=False
                )["purchase_count"].sum()
            if not category_totals.empty:
                connection.executemany(
                    "INSERT INTO global_category_stats VALUES (?, ?, ?, ?)",
                    [
                        (cutoff, str(row.category_type), int(row.category_id), int(row.purchase_count))
                        for row in category_totals.itertuples(index=False)
                    ],
                )
        connection.commit()
        connection.execute("DROP TABLE product_events")
        connection.execute("DROP TABLE category_events")
        connection.commit()
        connection.close()
        os.replace(temporary_path, output_path)
        return output_path
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


@dataclass
class HistoryStore:
    """History API supporting both fixture frames and a persisted index."""

    orders: pd.DataFrame
    prior_products: Optional[pd.DataFrame] = None
    products: Optional[pd.DataFrame] = None
    aisles: Optional[pd.DataFrame] = None
    departments: Optional[pd.DataFrame] = None
    index: Optional[HistoryIndex] = None

    def __post_init__(self) -> None:
        self.orders = self.orders.copy().reset_index(drop=True)
        _require_non_empty(self.orders, "orders")
        _require_columns(self.orders, ORDER_COLUMNS, "orders")
        if self.prior_products is None and self.index is None:
            raise HistoryDataError("either prior_products or index is required")
        if self.prior_products is not None:
            self.prior_products = self.prior_products.copy().reset_index(drop=True)
            _require_non_empty(self.prior_products, "prior_products")
            _require_columns(self.prior_products, TRANSACTION_COLUMNS, "prior_products")
            if self.prior_products.duplicated(["order_id", "product_id"]).any():
                raise HistoryDataError("prior_products contains duplicate order/product rows")
            if not self.prior_products["reordered"].isin([0, 1]).all():
                raise HistoryDataError("prior_products.reordered must contain only 0 or 1")
            if not set(self.prior_products["order_id"]).issubset(
                set(self.orders.loc[self.orders["eval_set"] == "prior", "order_id"])
            ):
                raise HistoryDataError("prior_products contains order IDs that are not prior orders")
        if self.orders["order_id"].duplicated().any():
            raise HistoryDataError("orders contains duplicate order_id values")
        if self.products is not None:
            _require_columns(self.products, ["product_id", "product_name", "aisle_id", "department_id"], "products")
            if self.products["product_id"].duplicated().any():
                raise HistoryDataError("products contains duplicate product_id values")
        if self.aisles is not None:
            _require_columns(self.aisles, ["aisle_id", "aisle"], "aisles")
        if self.departments is not None:
            _require_columns(self.departments, ["department_id", "department"], "departments")

    @property
    def is_indexed(self) -> bool:
        return self.index is not None

    def close(self) -> None:
        if self.index is not None:
            self.index.close()

    def __enter__(self) -> "HistoryStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @classmethod
    def from_csv(
        cls, raw_dir: Path, *, index_path: Optional[Path] = None,
        chunk_size: int = 1_000_000, progress: bool = True,
    ) -> "HistoryStore":
        """Load catalogs and build/open an indexed prior-history representation."""
        raw_dir = Path(raw_dir)
        if index_path is None:
            index_path = raw_dir.parent.parent / "processed" / "history.sqlite"
        orders = pd.read_csv(raw_dir / "orders.csv", usecols=ORDER_COLUMNS)
        products = pd.read_csv(raw_dir / "products.csv")
        aisles = pd.read_csv(raw_dir / "aisles.csv")
        departments = pd.read_csv(raw_dir / "departments.csv")
        index_path = Path(index_path)
        if not index_path.exists():
            prepare_history_index(raw_dir, index_path, chunk_size=chunk_size, progress=progress)
        index = HistoryIndex(index_path)
        try:
            index.validate(raw_dir)
        except Exception:
            index.close()
            raise
        return cls(orders, None, products, aisles, departments, index)

    def target_order(self, target_order_id: int) -> pd.Series:
        matches = self.orders.loc[self.orders["order_id"] == target_order_id]
        if len(matches) != 1:
            raise HistoryDataError(f"target_order_id {target_order_id} was not found exactly once")
        return matches.iloc[0]

    def historical_orders(self, customer_id: int, target_order_number: int) -> pd.DataFrame:
        if self.index is not None:
            return self.index.orders_before(customer_id, target_order_number)
        return self.orders.loc[
            (self.orders["user_id"] == customer_id)
            & (self.orders["eval_set"] == "prior")
            & (self.orders["order_number"] < target_order_number)
        ].sort_values(["order_number", "order_id"]).reset_index(drop=True)

    def history_for_target(
        self, customer_id: int, target_order_id: int, target_order_number: Optional[int] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        target = self.target_order(target_order_id)
        if int(target["user_id"]) != int(customer_id):
            raise HistoryDataError("target_order_id does not belong to customer_id")
        cutoff = int(target["order_number"] if target_order_number is None else target_order_number)
        if cutoff != int(target["order_number"]):
            raise HistoryDataError("target_order_number does not match target_order_id")
        orders = self.historical_orders(customer_id, cutoff)
        if self.index is not None:
            transactions = self.index.transactions_for_customer_before(customer_id, cutoff)
        else:
            transactions = self.prior_products.loc[
                self.prior_products["order_id"].isin(set(orders["order_id"]))
            ].merge(
                orders[["order_id", "user_id", "order_number", "days_since_prior_order"]],
                on="order_id", how="inner", validate="many_to_one",
            )
        return orders, transactions.sort_values(["order_number", "order_id", "product_id"]).reset_index(drop=True)

    def transactions_before_order_number(self, target_order_number: int) -> pd.DataFrame:
        """Compatibility method; indexed feature paths use aggregate methods."""
        if self.index is not None:
            return pd.read_sql_query(
                """
                SELECT t.*, o.user_id, o.order_number, o.days_since_prior_order
                FROM prior_transactions AS t JOIN orders AS o ON o.order_id = t.order_id
                WHERE o.eval_set = 'prior' AND o.order_number < ?
                ORDER BY o.order_number, t.order_id, t.product_id
                """,
                self.index.connection, params=(target_order_number,),
            )
        prior_orders = self.orders.loc[
            (self.orders["eval_set"] == "prior")
            & (self.orders["order_number"] < target_order_number),
            ["order_id", "user_id", "order_number", "days_since_prior_order"],
        ]
        return self.prior_products.merge(prior_orders, on="order_id", how="inner", validate="many_to_one")

    def target_products(self, target_order_id: int) -> pd.DataFrame:
        if self.index is not None:
            return self.index.transactions_for_order(target_order_id)
        return self.prior_products.loc[
            self.prior_products["order_id"] == target_order_id,
            ["order_id", "product_id", "reordered"],
        ].copy()

    def prior_orders_for_customer(self, customer_id: int) -> pd.DataFrame:
        return self.historical_orders(customer_id, int(self.orders["order_number"].max()) + 1)

    def product_statistics(
        self, target_order_number: int, product_ids: Optional[Sequence[int]] = None
    ) -> pd.DataFrame:
        if self.index is not None:
            return self.index.product_statistics(target_order_number, product_ids)
        transactions = self.transactions_before_order_number(target_order_number)
        result = transactions.groupby("product_id").agg(
            product_global_purchase_count=("order_id", "size"),
            product_global_reorder_count=("reordered", "sum"),
        ).reset_index()
        if product_ids is not None:
            result = result.loc[result["product_id"].isin(set(product_ids))]
        return result

    def top_products(
        self, target_order_number: int, limit: int,
        departments: Optional[set[int]] = None, aisles: Optional[set[int]] = None,
    ) -> pd.DataFrame:
        if self.index is not None:
            return self.index.top_products(target_order_number, limit, departments, aisles)
        result = self.product_statistics(target_order_number).merge(
            self.products[["product_id", "aisle_id", "department_id"]],
            on="product_id", how="left", validate="one_to_one",
        )
        if departments or aisles:
            result = result.loc[
                result["department_id"].isin(departments or set())
                | result["aisle_id"].isin(aisles or set())
            ]
        return result.assign(score=result["product_global_purchase_count"]).sort_values(
            ["score", "product_id"], ascending=[False, True]
        ).head(limit)

    def category_statistics(self, target_order_number: int) -> pd.DataFrame:
        if self.index is not None:
            return self.index.category_statistics(target_order_number)
        transactions = self.transactions_before_order_number(target_order_number).merge(
            self.products[["product_id", "aisle_id", "department_id"]],
            on="product_id", how="left", validate="many_to_one",
        )
        department = transactions.groupby("department_id").size().rename("purchase_count").reset_index()
        department["category_type"] = "department"
        department = department.rename(columns={"department_id": "category_id"})
        aisle = transactions.groupby("aisle_id").size().rename("purchase_count").reset_index()
        aisle["category_type"] = "aisle"
        aisle = aisle.rename(columns={"aisle_id": "category_id"})
        return pd.concat([department, aisle], ignore_index=True)

    def catalog(self) -> pd.DataFrame:
        if self.products is None:
            raise HistoryDataError("products catalog is required for personalized features")
        catalog = self.products.copy()
        if self.departments is not None:
            catalog = catalog.merge(
                self.departments, on="department_id", how="left", validate="many_to_one"
            )
        return catalog
