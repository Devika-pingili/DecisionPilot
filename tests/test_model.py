import math
import unittest

import pandas as pd

from backend.ml.dataset import build_development_dataset
from backend.ml.model import (
    DEFAULT_K_VALUES,
    IDENTIFIER_COLUMNS,
    compute_grouped_ranking_metrics,
    fit_ranking_model,
    get_model_feature_columns,
    rank_validation_predictions,
)
from backend.features.history import HistoryStore
from tests.feature_fixture import make_fixture


class RankingModelTests(unittest.TestCase):
    def setUp(self):
        frames = make_fixture()
        self.orders, self.prior, self.train_products, self.products, self.aisles, self.departments = frames
        self.history = HistoryStore(
            self.orders, self.prior, self.products, self.aisles, self.departments
        )
        self.result = build_development_dataset(self.history, customer_limit=2)

    def test_model_features_exclude_identifiers_and_text(self):
        features = get_model_feature_columns(self.result.train)
        self.assertNotIn("customer_id", features)
        self.assertNotIn("target_order_id", features)
        self.assertNotIn("target_order_number", features)
        self.assertNotIn("candidate_product_id", features)
        self.assertNotIn("product_id", features)
        self.assertNotIn("product_name", features)
        self.assertIn("customer_product_purchase_count", features)
        self.assertTrue(features)

    def test_model_trains_on_the_fixture_data(self):
        artifacts = fit_ranking_model(self.result.train, self.result.validation, random_state=42)
        self.assertEqual(len(artifacts["training_rows"]), len(self.result.train))
        self.assertEqual(len(artifacts["validation_predictions"]), len(self.result.validation))
        self.assertEqual(artifacts["metadata"]["training_row_count"], len(self.result.train))
        self.assertEqual(artifacts["metadata"]["validation_row_count"], len(self.result.validation))

    def test_scores_are_numeric_and_finite(self):
        artifacts = fit_ranking_model(self.result.train, self.result.validation, random_state=42)
        scores = artifacts["validation_predictions"]["model_score"]
        self.assertTrue(pd.api.types.is_numeric_dtype(scores))
        self.assertTrue((pd.notna(scores)).all())
        self.assertTrue((scores.map(lambda value: math.isfinite(float(value)))).all())

    def test_grouped_ranking_uses_model_score_then_product_id(self):
        predictions = pd.DataFrame([
            {"customer_id": 1, "target_order_id": 101, "candidate_product_id": 8, "model_score": 0.9, "y": 0},
            {"customer_id": 1, "target_order_id": 101, "candidate_product_id": 3, "model_score": 0.9, "y": 1},
            {"customer_id": 1, "target_order_id": 101, "candidate_product_id": 5, "model_score": 0.7, "y": 0},
            {"customer_id": 2, "target_order_id": 202, "candidate_product_id": 12, "model_score": 0.5, "y": 1},
        ])
        ranked = rank_validation_predictions(predictions)
        first_group = ranked.loc[ranked["customer_id"] == 1, "candidate_product_id"].tolist()
        self.assertEqual(first_group, [3, 8, 5])
        self.assertTrue(ranked["customer_id"].nunique() == 2)

    def test_validation_rows_are_never_used_for_fitting(self):
        artifacts = fit_ranking_model(self.result.train, self.result.validation, random_state=42)
        self.assertEqual(artifacts["model"].n_features_in_, len(artifacts["feature_columns"]))
        self.assertEqual(artifacts["metadata"]["training_row_count"], len(self.result.train))
        self.assertNotEqual(artifacts["metadata"]["training_row_count"], len(self.result.validation))

    def test_identifier_columns_are_not_model_features(self):
        features = get_model_feature_columns(self.result.train)
        for column in IDENTIFIER_COLUMNS:
            self.assertNotIn(column, features)

    def test_target_derived_columns_are_excluded(self):
        features = get_model_feature_columns(self.result.train)
        self.assertNotIn("product_name", features)
        self.assertNotIn("target_order_id", features)
        self.assertNotIn("customer_id", features)

    def test_empty_input_is_handled_clearly(self):
        empty = pd.DataFrame(columns=["customer_id", "target_order_id", "target_order_number", "candidate_product_id", "y"])
        with self.assertRaisesRegex(ValueError, "empty|No numeric"):
            fit_ranking_model(empty, empty, random_state=42)

    def test_ranking_metrics_are_reported_for_each_k(self):
        predictions = pd.DataFrame([
            {"customer_id": 1, "target_order_id": 10, "candidate_product_id": 2, "y": 1, "model_score": 0.9},
            {"customer_id": 1, "target_order_id": 10, "candidate_product_id": 4, "y": 0, "model_score": 0.6},
            {"customer_id": 1, "target_order_id": 10, "candidate_product_id": 1, "y": 1, "model_score": 0.8},
        ])
        metrics = compute_grouped_ranking_metrics(predictions, k_values=DEFAULT_K_VALUES)
        self.assertIn("precision_at_5", metrics)
        self.assertIn("recall_at_5", metrics)
        self.assertIn("f1_at_5", metrics)
        self.assertIn("map_at_5", metrics)
        self.assertIn("ndcg_at_5", metrics)


if __name__ == "__main__":
    unittest.main()
