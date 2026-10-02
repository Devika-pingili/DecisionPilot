import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd

from backend.api.main import create_app, DEFAULT_INDEX_PATH, DEFAULT_RAW_DIR
from backend.config import PROJECT_ROOT, resolve_data_paths
from backend.features.history import HistoryStore, SOURCE_FILES
from backend.recommendation.recommend import DEFAULT_MODEL_PATH

DEMO_ROOT = PROJECT_ROOT / "data" / "demo"
EXPECTED_DEMO_FILES = {
    *(f"raw/instacart/{name}" for name in SOURCE_FILES),
    "processed/history.sqlite",
    "processed/ml_dev/model.joblib",
}


class DemoDataConfigurationTests(unittest.TestCase):
    def test_default_configuration_keeps_full_local_paths(self):
        with patch.dict(os.environ, {}, clear=True):
            paths = resolve_data_paths()

        self.assertEqual(paths.root, PROJECT_ROOT / "data")
        self.assertEqual(paths.raw_dir, DEFAULT_RAW_DIR)
        self.assertEqual(paths.index_path, DEFAULT_INDEX_PATH)
        self.assertEqual(paths.model_path, DEFAULT_MODEL_PATH)

    def test_demo_environment_resolves_demo_artifact_paths(self):
        with patch.dict(os.environ, {"DECISIONPILOT_DATA_ROOT": "data/demo"}, clear=True):
            paths = resolve_data_paths()

        self.assertEqual(paths.root, DEMO_ROOT)
        self.assertEqual(paths.raw_dir, DEMO_ROOT / "raw" / "instacart")
        self.assertEqual(paths.index_path, DEMO_ROOT / "processed" / "history.sqlite")
        self.assertEqual(paths.model_path, DEMO_ROOT / "processed" / "ml_dev" / "model.joblib")

    def test_demo_artifact_tree_contains_only_bounded_runtime_files(self):
        self.assertTrue(DEMO_ROOT.is_dir())
        actual_files = {
            path.relative_to(DEMO_ROOT).as_posix()
            for path in DEMO_ROOT.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, EXPECTED_DEMO_FILES)
        self.assertLess(
            (DEMO_ROOT / "processed" / "history.sqlite").stat().st_size,
            100 * 1024 * 1024,
        )
        self.assertFalse((DEMO_ROOT / "raw" / "instacart" / "order_products__train.csv").exists())

    def test_demo_sources_are_internally_consistent(self):
        raw_dir = DEMO_ROOT / "raw" / "instacart"
        orders = pd.read_csv(raw_dir / "orders.csv")
        prior_ids = set(
            orders.loc[orders["eval_set"] == "prior", "order_id"].astype(int)
        )
        transactions = pd.read_csv(raw_dir / "order_products__prior.csv")
        products = pd.read_csv(raw_dir / "products.csv")
        aisles = pd.read_csv(raw_dir / "aisles.csv")
        departments = pd.read_csv(raw_dir / "departments.csv")

        self.assertTrue(set(transactions["order_id"].astype(int)).issubset(prior_ids))
        self.assertTrue(
            set(transactions["product_id"].astype(int)).issubset(
                set(products["product_id"].astype(int))
            )
        )
        self.assertTrue(set(products["aisle_id"]).issubset(set(aisles["aisle_id"])))
        self.assertTrue(
            set(products["department_id"]).issubset(set(departments["department_id"]))
        )

    def test_fresh_demo_root_builds_only_its_own_index(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "demo"
            shutil.copytree(DEMO_ROOT / "raw", data_root / "raw")
            model_path = data_root / "processed" / "ml_dev" / "model.joblib"
            model_path.parent.mkdir(parents=True)
            shutil.copy2(DEMO_ROOT / "processed" / "ml_dev" / "model.joblib", model_path)

            with patch.dict(os.environ, {"DECISIONPILOT_DATA_ROOT": str(data_root)}):
                with TestClient(create_app()) as client:
                    self.assertEqual(client.get("/health").json(), {"status": "healthy"})
                    recommendations = client.get(
                        "/customers/1/recommendations?top_k=5&order_number=10"
                    )

            self.assertEqual(recommendations.status_code, 200)
            self.assertTrue((data_root / "processed" / "history.sqlite").is_file())
            self.assertLess(
                (data_root / "processed" / "history.sqlite").stat().st_size,
                100 * 1024 * 1024,
            )

    def test_demo_history_opens_and_contains_customer_one(self):
        paths = resolve_data_paths(DEMO_ROOT)
        history = HistoryStore.from_csv(paths.raw_dir, index_path=paths.index_path, progress=False)
        try:
            self.assertTrue((history.orders["user_id"] == 1).any())
            self.assertGreaterEqual(len(history.prior_orders_for_customer(1)), 3)
        finally:
            history.close()

    def test_demo_customer_one_summary_and_recommendations_work(self):
        with patch.dict(os.environ, {"DECISIONPILOT_DATA_ROOT": "data/demo"}):
            with TestClient(create_app()) as client:
                summary = client.get("/customers/1/summary")
                recommendations = client.get(
                    "/customers/1/recommendations?top_k=5&order_number=10"
                )

        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["customer_id"], 1)
        self.assertGreaterEqual(summary.json()["total_orders"], 3)
        self.assertEqual(recommendations.status_code, 200)
        payload = recommendations.json()
        self.assertEqual(payload["customer_id"], 1)
        self.assertEqual(len(payload["recommendations"]), 5)
        self.assertEqual(
            set(payload["recommendations"][0]),
            {
                "rank",
                "product_id",
                "product_name",
                "department_id",
                "aisle_id",
                "model_score",
                "explanation_short",
                "explanation_reason_codes",
            },
        )
        self.assertTrue(all(item["product_name"] for item in payload["recommendations"]))


if __name__ == "__main__":
    unittest.main()
