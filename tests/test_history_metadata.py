import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backend.features.history import (
    PREPARATION_CONFIG,
    HistoryIndex,
    HistoryStore,
    IndexCompatibilityError,
    prepare_history_index,
)
from tests.feature_fixture import make_fixture


class HistoryMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.raw_dir = self.root / "raw" / "instacart"
        self.raw_dir.mkdir(parents=True)
        orders, prior, train, products, aisles, departments = make_fixture()
        orders.to_csv(self.raw_dir / "orders.csv", index=False)
        prior.to_csv(self.raw_dir / "order_products__prior.csv", index=False)
        train.to_csv(self.raw_dir / "order_products__train.csv", index=False)
        products.to_csv(self.raw_dir / "products.csv", index=False)
        aisles.to_csv(self.raw_dir / "aisles.csv", index=False)
        departments.to_csv(self.raw_dir / "departments.csv", index=False)
        self.index_path = self.root / "processed" / "history.sqlite"
        prepare_history_index(self.raw_dir, self.index_path, chunk_size=2, progress=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_index_validates(self):
        with HistoryStore.from_csv(self.raw_dir, index_path=self.index_path, progress=False) as history:
            self.assertTrue(history.index.validate(self.raw_dir))

    def test_metadata_contains_versions_sources_and_configuration(self):
        connection = sqlite3.connect(self.index_path)
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        finally:
            connection.close()
        self.assertIn("index_format_version", metadata)
        self.assertIn("preparation_schema_version", metadata)
        self.assertEqual(json.loads(metadata["source_files"]), [
            "orders.csv", "order_products__prior.csv", "products.csv",
            "aisles.csv", "departments.csv",
        ])
        self.assertEqual(json.loads(metadata["preparation_config"]), PREPARATION_CONFIG)

    def test_changed_source_is_rejected(self):
        orders_path = self.raw_dir / "orders.csv"
        orders = pd.read_csv(orders_path)
        orders.loc[0, "order_dow"] = 6
        orders.to_csv(orders_path, index=False)

        with self.assertRaisesRegex(IndexCompatibilityError, "source metadata mismatch"):
            HistoryStore.from_csv(self.raw_dir, index_path=self.index_path, progress=False)

    def test_changed_index_version_is_rejected(self):
        with HistoryIndex(self.index_path) as index:
            with patch("backend.features.history.INDEX_FORMAT_VERSION", 999):
                with self.assertRaisesRegex(IndexCompatibilityError, "format version"):
                    index.validate()

    def test_missing_metadata_is_rejected(self):
        connection = sqlite3.connect(self.index_path)
        try:
            connection.execute("DELETE FROM metadata")
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(IndexCompatibilityError, "missing required metadata"):
            HistoryIndex(self.index_path)

    def test_missing_table_is_rejected(self):
        path = self.root / "missing-table.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(IndexCompatibilityError, "missing required tables"):
            HistoryIndex(path)

    def test_changed_configuration_is_rejected(self):
        changed = dict(PREPARATION_CONFIG)
        changed["cutoff_rule"] = "order_number <= target_order_number"
        with HistoryIndex(self.index_path) as index:
            with self.assertRaisesRegex(IndexCompatibilityError, "configuration"):
                index.validate(expected_config=changed)


if __name__ == "__main__":
    unittest.main()
