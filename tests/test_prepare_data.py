import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backend import prepare_data


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.raw_dir = root / "raw" / "instacart"
        self.processed_dir = root / "processed"
        self.raw_dir.mkdir(parents=True)
        self.processed_dir.mkdir()
        self.patches = ExitStack()
        self.patches.enter_context(patch.object(prepare_data, "RAW_DIR", self.raw_dir))
        self.patches.enter_context(patch.object(prepare_data, "PROCESSED_DIR", self.processed_dir))
        self.patches.enter_context(patch.object(prepare_data, "CHUNK_SIZE", 1))

    def tearDown(self):
        self.patches.close()
        self.temp_dir.cleanup()

    def write_valid_inputs(self):
        pd.DataFrame([
            [1, 10, 1, 0, 10, None, "prior"],
            [2, 10, 2, 1, 11, 7, "prior"],
        ], columns=[
            "order_id", "user_id", "order_number", "order_dow",
            "order_hour_of_day", "days_since_prior_order", "eval_set",
        ]).to_csv(self.raw_dir / "orders.csv", index=False)
        pd.DataFrame([
            [1, 1, 1, 0],
            [1, 2, 2, 1],
            [2, 1, 1, 1],
        ], columns=["order_id", "product_id", "add_to_cart_order", "reordered"]).to_csv(
            self.raw_dir / "order_products__prior.csv", index=False
        )
        pd.DataFrame([
            [1, "Apple", 1, 1],
            [2, "Bread", 1, 1],
        ], columns=["product_id", "product_name", "aisle_id", "department_id"]).to_csv(
            self.raw_dir / "products.csv", index=False
        )
        pd.DataFrame([[1, "food"]], columns=["department_id", "department"]).to_csv(
            self.raw_dir / "departments.csv", index=False
        )

    def run_main(self):
        with self.assertRaises(SystemExit) as raised:
            prepare_data.main()
        return raised.exception.code

    def test_valid_input_generates_all_outputs(self):
        self.write_valid_inputs()

        with patch.object(sys, "exit", side_effect=AssertionError("unexpected exit")):
            prepare_data.main()

        expected = {
            "order_features.csv",
            "product_features.csv",
            "department_features.csv",
            "order_time_patterns.csv",
        }
        self.assertEqual({path.name for path in self.processed_dir.iterdir()}, expected)
        order_features = pd.read_csv(self.processed_dir / "order_features.csv")
        self.assertEqual(len(order_features), 2)
        self.assertEqual(int(order_features.loc[0, "basket_size"]), 2)

    def test_empty_input_fails(self):
        self.write_valid_inputs()
        (self.raw_dir / "order_products__prior.csv").write_text("")

        self.assertEqual(self.run_main(), 1)
        self.assertEqual(list(self.processed_dir.iterdir()), [])

    def test_missing_required_column_fails(self):
        self.write_valid_inputs()
        orders = pd.read_csv(self.raw_dir / "orders.csv").drop(columns=["user_id"])
        orders.to_csv(self.raw_dir / "orders.csv", index=False)

        self.assertEqual(self.run_main(), 1)

    def test_duplicate_ids_fail(self):
        self.write_valid_inputs()
        orders = pd.read_csv(self.raw_dir / "orders.csv")
        orders.loc[1, "order_id"] = orders.loc[0, "order_id"]
        orders.to_csv(self.raw_dir / "orders.csv", index=False)

        self.assertEqual(self.run_main(), 1)

    def test_invalid_relationship_fails(self):
        self.write_valid_inputs()
        products = pd.read_csv(self.raw_dir / "products.csv")
        products.loc[0, "department_id"] = 99
        products.to_csv(self.raw_dir / "products.csv", index=False)

        self.assertEqual(self.run_main(), 1)

    def test_missing_transaction_order_join_fails(self):
        self.write_valid_inputs()
        prior = pd.read_csv(self.raw_dir / "order_products__prior.csv")
        prior.loc[0, "order_id"] = 999
        prior.to_csv(self.raw_dir / "order_products__prior.csv", index=False)

        self.assertEqual(self.run_main(), 1)

    def test_chunk_boundary_aggregation(self):
        self.write_valid_inputs()

        orders, products = prepare_data.scan_prior_transactions()

        order_one = orders.loc[orders["order_id"] == 1].iloc[0]
        self.assertEqual(int(order_one["basket_size"]), 2)
        self.assertEqual(int(order_one["max_add_to_cart_order"]), 2)
        product_one = products.loc[products["product_id"] == 1].iloc[0]
        self.assertEqual(int(product_one["purchase_count"]), 2)
        self.assertEqual(int(product_one["reorder_count"]), 1)

    def test_output_validation_failure_exits_nonzero(self):
        self.write_valid_inputs()
        with patch.object(prepare_data, "validate_outputs", return_value=False):
            self.assertEqual(self.run_main(), 1)
        self.assertEqual(list(self.processed_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
