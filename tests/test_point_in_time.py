import unittest

import pandas as pd

from backend.features.candidates import generate_candidates
from backend.features.history import HistoryStore
from backend.features.labels import build_labels, build_rolling_training_examples
from backend.features.point_in_time import FEATURE_COLUMNS, build_point_in_time_features
from tests.feature_fixture import make_fixture


class PointInTimeTests(unittest.TestCase):
    def setUp(self):
        frames = make_fixture()
        self.orders, self.prior_products, self.train_products = frames[:3]
        self.history = HistoryStore(
            self.orders, self.prior_products, frames[3], frames[4], frames[5]
        )
        self.candidates = generate_candidates(self.history, 10, 104, 4, candidate_limit=20)

    def test_feature_schema_and_historical_values(self):
        features = build_point_in_time_features(self.history, self.candidates)
        product_one = features.loc[features["candidate_product_id"] == 1].iloc[0]
        product_five = pd.DataFrame([{
            "customer_id": 10,
            "target_order_id": 104,
            "target_order_number": 4,
            "candidate_product_id": 5,
        }])
        product_five_features = build_point_in_time_features(self.history, product_five).iloc[0]

        self.assertEqual(features.columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(int(product_one["customer_product_purchase_count"]), 2)
        self.assertEqual(int(product_one["customer_product_reorder_count"]), 1)
        self.assertAlmostEqual(product_one["customer_product_reorder_rate"], 0.5)
        self.assertEqual(int(product_one["orders_since_last_product_purchase"]), 1)
        self.assertEqual(product_one["days_since_last_product_purchase"], 3.0)
        self.assertEqual(int(product_one["customer_total_orders"]), 3)
        self.assertAlmostEqual(product_one["customer_average_basket_size"], 2.0)
        self.assertAlmostEqual(product_one["customer_recent_basket_size"], 2.0)
        self.assertEqual(int(product_five_features["customer_product_purchase_count"]), 0)
        self.assertTrue(pd.isna(product_five_features["customer_product_reorder_rate"]))
        self.assertTrue(pd.isna(product_five_features["orders_since_last_product_purchase"]))

    def test_target_interval_is_excluded(self):
        target = pd.DataFrame([{
            "customer_id": 10,
            "target_order_id": 104,
            "target_order_number": 4,
            "candidate_product_id": 1,
        }])
        feature = build_point_in_time_features(self.history, target).iloc[0]

        self.assertEqual(feature["days_since_last_product_purchase"], 3.0)

    def test_future_order_cannot_change_earlier_features(self):
        target = pd.DataFrame([{
            "customer_id": 10,
            "target_order_id": 103,
            "target_order_number": 3,
            "candidate_product_id": 3,
        }])
        before = build_point_in_time_features(self.history, target).iloc[0]
        future_orders = self.orders.copy()
        future_orders.loc[future_orders["order_id"] == 105, "order_number"] = 2
        altered = HistoryStore(
            future_orders, self.prior_products, self.history.products,
            self.history.aisles, self.history.departments
        )
        after = build_point_in_time_features(altered, target).iloc[0]

        self.assertEqual(before["customer_product_purchase_count"], after["customer_product_purchase_count"])
        self.assertEqual(before["product_global_purchase_count"], after["product_global_purchase_count"])

    def test_train_target_rows_do_not_change_features(self):
        target = self.candidates.iloc[[0]].copy()
        before = build_point_in_time_features(self.history, target)
        changed_target_products = self.train_products.copy()
        changed_target_products.loc[:, "product_id"] = 5
        after = build_point_in_time_features(self.history, target)

        pd.testing.assert_frame_equal(before, after)
        self.assertFalse(changed_target_products.empty)

    def test_rolling_examples_use_each_prior_order_as_target(self):
        rolling_orders = self.orders.copy()
        rolling_orders.loc[rolling_orders["order_id"] == 104, "eval_set"] = "prior"
        rolling_products = pd.concat([
            self.prior_products,
            pd.DataFrame([[104, 1, 1, 1]], columns=self.prior_products.columns),
        ], ignore_index=True)
        rolling_history = HistoryStore(
            rolling_orders, rolling_products, self.history.products,
            self.history.aisles, self.history.departments
        )
        examples = build_rolling_training_examples(rolling_history, candidate_limit=20)
        target_ids = set(examples["target_order_id"])

        self.assertEqual(target_ids, {102, 103, 104})
        self.assertFalse(examples.duplicated([
            "customer_id", "target_order_id", "target_order_number", "candidate_product_id"
        ]).any())

    def test_global_statistics_change_at_explicit_cutoffs(self):
        early_target = pd.DataFrame([{
            "customer_id": 10,
            "target_order_id": 102,
            "target_order_number": 2,
            "candidate_product_id": 3,
        }])
        late_target = pd.DataFrame([{
            "customer_id": 10,
            "target_order_id": 104,
            "target_order_number": 4,
            "candidate_product_id": 3,
        }])
        early = build_point_in_time_features(self.history, early_target).iloc[0]
        late = build_point_in_time_features(self.history, late_target).iloc[0]

        self.assertEqual(early["product_global_purchase_count"], 0)
        self.assertEqual(late["product_global_purchase_count"], 1)

    def test_labels_and_zero_division(self):
        target = self.candidates.copy()
        labels = build_labels(target, self.train_products)
        self.assertEqual(set(labels["y"]), {0, 1})

        never_purchased = pd.DataFrame([{
            "customer_id": 10,
            "target_order_id": 104,
            "target_order_number": 4,
            "candidate_product_id": 5,
        }])
        feature = build_point_in_time_features(self.history, never_purchased).iloc[0]
        self.assertTrue(pd.isna(feature["product_historical_reorder_rate"]))


if __name__ == "__main__":
    unittest.main()
