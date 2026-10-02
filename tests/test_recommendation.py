import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.features.history import HistoryStore
from backend.recommendation.recommend import (
    DEFAULT_MODEL_PATH,
    RECOMMENDATION_COLUMNS,
    explain_historical_features,
    load_model,
    recommend_products,
    write_recommendations,
)
from tests.feature_fixture import make_fixture


class ConstantScoreModel:
    classes_ = np.array([0, 1])

    def __init__(self, feature_names):
        self.feature_names_in_ = np.array(feature_names, dtype=object)

    def predict_proba(self, frame):
        return np.tile(np.array([[0.5, 0.5]]), (len(frame), 1))


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_model = load_model(DEFAULT_MODEL_PATH)

    def setUp(self):
        frames = make_fixture()
        self.orders, self.prior, self.train_products, self.products, self.aisles, self.departments = frames
        self.history = HistoryStore(
            self.orders, self.prior, self.products, self.aisles, self.departments
        )

    def _recommend(self, **kwargs):
        return recommend_products(
            self.history,
            10,
            103,
            3,
            model=self.saved_model,
            **kwargs,
        )

    def test_recommendation_output_schema(self):
        result = self._recommend()
        self.assertEqual(result.columns.tolist(), RECOMMENDATION_COLUMNS)
        self.assertFalse(result.empty)

    def test_ranking_is_score_descending(self):
        result = self._recommend()
        self.assertTrue(result["model_score"].is_monotonic_decreasing)

    def test_product_id_is_deterministic_score_tie_breaker(self):
        constant_model = ConstantScoreModel(self.saved_model.feature_names_in_)
        result = recommend_products(self.history, 10, 103, 3, model=constant_model)
        self.assertEqual(result["product_id"].tolist(), sorted(result["product_id"].tolist()))

    def test_no_duplicate_customer_context_product_rows(self):
        result = self._recommend()
        self.assertFalse(result.duplicated(["customer_id", "target_order_id", "product_id"]).any())

    def test_top_k_limit_works(self):
        result = self._recommend(top_k=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["rank"].tolist(), [1, 2, 3])

    def test_explanations_are_deterministic_and_evidence_based(self):
        evidence = {
            "customer_product_purchase_count": 2,
            "customer_recent_product_flag": 1,
            "customer_product_reorder_count": 1,
            "customer_department_affinity": 0.25,
            "customer_aisle_affinity": 0.1,
            "product_global_purchase_count": 10,
        }
        first = explain_historical_features(evidence)
        self.assertEqual(first, explain_historical_features(evidence))
        self.assertEqual(
            first["explanation_reason_codes"],
            "CUSTOMER_FREQUENT|CUSTOMER_RECENT|CUSTOMER_REORDER|DEPARTMENT_AFFINITY|AISLE_AFFINITY|GLOBAL_POPULARITY",
        )
        no_evidence = explain_historical_features({
            "customer_product_purchase_count": 0,
            "customer_recent_product_flag": 0,
            "customer_product_reorder_count": 0,
            "customer_department_affinity": 0,
            "customer_aisle_affinity": 0,
            "product_global_purchase_count": 0,
        })
        self.assertEqual(no_evidence["explanation_reason_codes"], "")

    def test_target_and_future_products_do_not_change_as_of_recommendations(self):
        altered_prior = self.prior.copy()
        target_rows = altered_prior["order_id"] == 102
        future_rows = altered_prior["order_id"] == 103
        altered_prior.loc[target_rows, "product_id"] = [2, 5]
        altered_prior.loc[future_rows, "product_id"] = [1, 3]
        altered_history = HistoryStore(
            self.orders, altered_prior, self.products, self.aisles, self.departments
        )
        original = recommend_products(self.history, 10, 102, 2, model=self.saved_model)
        altered = recommend_products(altered_history, 10, 102, 2, model=self.saved_model)
        pd.testing.assert_frame_equal(original, altered)

    def test_saved_model_loads_and_scores_live_candidates(self):
        model = load_model(DEFAULT_MODEL_PATH)
        result = recommend_products(self.history, 10, 103, 3, model=model)
        self.assertEqual(len(result), len(self._recommend()))
        self.assertTrue(np.isfinite(result["model_score"].to_numpy()).all())

    def test_first_order_context_returns_empty_schema(self):
        result = recommend_products(self.history, 10, 101, 1, model=self.saved_model)
        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), RECOMMENDATION_COLUMNS)

    def test_malformed_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "belong"):
            recommend_products(self.history, 20, 103, model=self.saved_model)
        with self.assertRaisesRegex(ValueError, "does not match"):
            recommend_products(self.history, 10, 103, 2, model=self.saved_model)
        with self.assertRaisesRegex(ValueError, "integers"):
            recommend_products(self.history, "customer", 103, model=self.saved_model)
        with self.assertRaisesRegex(ValueError, "positive"):
            self._recommend(top_k=0)

    def test_non_prior_prediction_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prior order"):
            recommend_products(self.history, 10, 104, 4, model=self.saved_model)

    def test_output_writer_is_byte_deterministic(self):
        recommendations = self._recommend(top_k=5)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_path = write_recommendations(recommendations, Path(first_dir) / "recommendations.csv")
            second_path = write_recommendations(recommendations, Path(second_dir) / "recommendations.csv")
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

    def test_output_writer_rejects_malformed_schema(self):
        with self.assertRaisesRegex(ValueError, "missing output columns"):
            write_recommendations(pd.DataFrame(), Path("unused.csv"))


if __name__ == "__main__":
    unittest.main()