import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.features.candidates import generate_candidates
from backend.features.history import HistoryStore
from backend.ml.dataset import DATASET_COLUMNS, build_development_dataset
from tests.feature_fixture import make_fixture


class MlDatasetTests(unittest.TestCase):
    def setUp(self):
        frames = make_fixture()
        self.orders, self.prior, self.train_products, self.products, self.aisles, self.departments = frames
        self.history = HistoryStore(
            self.orders, self.prior, self.products, self.aisles, self.departments
        )

    def test_training_row_schema_and_labels(self):
        result = build_development_dataset(self.history, customer_limit=2)

        self.assertEqual(result.train.columns.tolist(), DATASET_COLUMNS)
        self.assertEqual(result.validation.columns.tolist(), DATASET_COLUMNS)
        self.assertIn(1, set(result.train["y"]))
        self.assertIn(0, set(result.train["y"]))
        self.assertIn(1, set(result.validation["y"]))
        self.assertIn(0, set(result.validation["y"]))
        self.assertNotIn("basket_size", DATASET_COLUMNS)
        self.assertNotIn("reordered_items", DATASET_COLUMNS)
        self.assertNotIn("reorder_rate", DATASET_COLUMNS)
        self.assertNotIn("max_add_to_cart_order", DATASET_COLUMNS)

    def test_rows_are_restricted_to_existing_candidates(self):
        result = build_development_dataset(self.history, customer_limit=2)
        combined = pd.concat([result.train, result.validation], ignore_index=True)
        for (customer_id, target_order_id, target_order_number), group in combined.groupby([
            "customer_id", "target_order_id", "target_order_number"
        ]):
            expected = generate_candidates(
                self.history,
                int(customer_id),
                int(target_order_id),
                int(target_order_number),
            )
            self.assertEqual(
                set(group["candidate_product_id"]),
                set(expected["candidate_product_id"]),
            )

    def test_target_and_future_orders_do_not_enter_features(self):
        result = build_development_dataset(self.history, customer_limit=2)
        train_product_one = result.train.loc[
            (result.train["target_order_id"] == 102)
            & (result.train["candidate_product_id"] == 1)
        ].iloc[0]
        validation_product_two = result.validation.loc[
            (result.validation["target_order_id"] == 103)
            & (result.validation["candidate_product_id"] == 2)
        ].iloc[0]

        self.assertEqual(train_product_one["customer_product_purchase_count"], 1)
        self.assertEqual(validation_product_two["customer_product_purchase_count"], 1)
        self.assertNotIn(103, result.train["target_order_id"].tolist())

    def test_train_validation_split_is_chronological_per_customer(self):
        result = build_development_dataset(self.history, customer_limit=2)
        train_targets = result.train[["customer_id", "target_order_number"]].drop_duplicates()
        validation_targets = result.validation[["customer_id", "target_order_number"]].drop_duplicates()
        for customer_id in set(train_targets["customer_id"]) & set(validation_targets["customer_id"]):
            train_max = train_targets.loc[
                train_targets["customer_id"] == customer_id, "target_order_number"
            ].max()
            validation_min = validation_targets.loc[
                validation_targets["customer_id"] == customer_id, "target_order_number"
            ].min()
            self.assertLess(train_max, validation_min)

    def test_grouping_and_positive_negative_counts(self):
        result = build_development_dataset(self.history, customer_limit=2)
        for frame in [result.train, result.validation]:
            self.assertFalse(frame.duplicated([
                "customer_id", "target_order_id", "target_order_number", "candidate_product_id"
            ]).any())
        self.assertEqual(result.metadata["target_group_count"], 2)
        self.assertEqual(
            result.metadata["positive_rows"] + result.metadata["negative_rows"],
            result.metadata["total_rows"],
        )

    def test_train_targets_are_not_read(self):
        changed_train = self.train_products.copy()
        changed_train["product_id"] = 5
        first = build_development_dataset(self.history, customer_limit=2)
        second = build_development_dataset(self.history, customer_limit=2)
        pd.testing.assert_frame_equal(first.train, second.train)
        pd.testing.assert_frame_equal(first.validation, second.validation)
        self.assertFalse(changed_train.empty)

    def test_empty_and_insufficient_history(self):
        result = build_development_dataset(self.history, customer_limit=2, min_history=3)

        self.assertTrue(result.train.empty)
        self.assertTrue(result.validation.empty)
        self.assertEqual(result.metadata["training_target_count"], 0)
        self.assertEqual(result.metadata["validation_target_count"], 0)

    def test_repeated_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = build_development_dataset(self.history, customer_limit=2, output_dir=Path(first_dir))
            second = build_development_dataset(self.history, customer_limit=2, output_dir=Path(second_dir))
            for filename in ["train.csv", "validation.csv", "candidate_diagnostics.csv", "metadata.json"]:
                first_hash = hashlib.sha256((Path(first_dir) / filename).read_bytes()).hexdigest()
                second_hash = hashlib.sha256((Path(second_dir) / filename).read_bytes()).hexdigest()
                self.assertEqual(first_hash, second_hash, filename)
            pd.testing.assert_frame_equal(first.train, second.train)
            pd.testing.assert_frame_equal(first.validation, second.validation)


if __name__ == "__main__":
    unittest.main()
