import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.features.history import HistoryStore
from backend.recommendation.recommend import DEFAULT_MODEL_PATH, load_model
from tests.feature_fixture import make_fixture


class RecommendationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model(DEFAULT_MODEL_PATH)

    def setUp(self):
        frames = make_fixture()
        self.orders, self.prior, self.train_products, self.products, self.aisles, self.departments = frames
        self.history = HistoryStore(
            self.orders, self.prior, self.products, self.aisles, self.departments
        )
        self.client = TestClient(create_app(history=self.history, model=self.model))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_root_and_health(self):
        self.assertEqual(
            self.client.get("/").json(),
            {"name": "DecisionPilot", "status": "ok", "version": "1.0"},
        )
        self.assertEqual(self.client.get("/health").json(), {"status": "healthy"})

    def test_ai_diagnostics_are_loopback_only_and_do_not_include_credentials(self):
        response = self.client.get("/diagnostics/ai")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload),
            {
                "api_key_configured",
                "model_configured",
                "model_configuration",
                "client_initialization",
                "provider_request_attempted",
                "last_error_type",
                "last_error_message",
            },
        )
        self.assertNotIn("api_key", payload)

    def test_cors_allows_only_configured_local_frontend_origins(self):
        allowed = self.client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(allowed.headers["access-control-allow-origin"], "http://localhost:5173")
        blocked = self.client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertNotIn("access-control-allow-origin", blocked.headers)

    def test_recommendations_schema_and_ranking(self):
        response = self.client.get("/customers/10/recommendations?top_k=5")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["customer_id"], 10)
        self.assertGreater(len(payload["recommendations"]), 0)
        self.assertLessEqual(len(payload["recommendations"]), 5)
        expected_fields = {
            "rank",
            "product_id",
            "product_name",
            "department_id",
            "aisle_id",
            "model_score",
            "explanation_short",
            "explanation_reason_codes",
        }
        self.assertEqual(set(payload["recommendations"][0]), expected_fields)
        self.assertEqual(
            [row["rank"] for row in payload["recommendations"]],
            list(range(1, len(payload["recommendations"]) + 1)),
        )
        ranked_keys = [
            (-row["model_score"], row["product_id"])
            for row in payload["recommendations"]
        ]
        self.assertEqual(ranked_keys, sorted(ranked_keys))
        self.assertIsInstance(payload["recommendations"][0]["explanation_reason_codes"], list)

    def test_explanation_endpoint_uses_server_evidence_and_preserves_recommendations(self):
        original = self.client.get("/customers/10/recommendations?top_k=5").json()
        selected = original["recommendations"][0]

        with patch(
            "backend.api.main.generate_ai_explanation",
            return_value="This product fits the verified evidence.",
        ) as explain:
            response = self.client.post(
                "/customers/10/recommendations/explain",
                json={
                    "product_id": selected["product_id"],
                    "top_k": 5,
                    "reason_codes": ["FAKE_REASON"],
                    "customer_history": "must not be forwarded",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["product_name"], selected["product_name"])
        self.assertEqual(payload["model_score"], selected["model_score"])
        self.assertEqual(payload["reason_codes"], selected["explanation_reason_codes"])
        self.assertEqual(payload["deterministic_explanation"], selected["explanation_short"])
        self.assertEqual(payload["ai_explanation"], "This product fits the verified evidence.")
        explain.assert_called_once_with(
            selected["product_name"], selected["explanation_reason_codes"]
        )
        self.assertEqual(
            self.client.get("/customers/10/recommendations?top_k=5").json(),
            original,
        )

    def test_explanation_unavailable_does_not_affect_recommendations(self):
        with patch("backend.ai.explanation.OpenAI") as provider:
            before = self.client.get("/customers/10/recommendations?top_k=5").json()
            selected = before["recommendations"][0]
            response = self.client.post(
                "/customers/10/recommendations/explain",
                json={"product_id": selected["product_id"], "top_k": 5},
            )
            after = self.client.get("/customers/10/recommendations?top_k=5").json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["available"])
        self.assertEqual(
            response.json()["message"], "AI explanation is currently unavailable."
        )
        self.assertEqual(response.json()["deterministic_explanation"], selected["explanation_short"])
        self.assertEqual(before, after)
        provider.assert_not_called()

    def test_explanation_endpoint_rejects_invalid_customer_context_and_unrecommended_product(self):
        invalid_customer = self.client.post(
            "/customers/999/recommendations/explain",
            json={"product_id": 1},
        )
        invalid_context = self.client.post(
            "/customers/10/recommendations/explain",
            json={"product_id": 1, "order_number": 99},
        )
        unranked_product = self.client.post(
            "/customers/10/recommendations/explain",
            json={"product_id": 999999, "top_k": 5},
        )

        self.assertEqual(invalid_customer.status_code, 404)
        self.assertEqual(invalid_context.status_code, 400)
        self.assertEqual(unranked_product.status_code, 404)

    def test_explanation_endpoint_validates_request_and_uses_requested_cutoff(self):
        recommendations = self.client.get(
            "/customers/10/recommendations?top_k=2&order_number=2"
        ).json()["recommendations"]
        selected = recommendations[0]

        invalid_product = self.client.post(
            "/customers/10/recommendations/explain",
            json={"product_id": 0, "top_k": 2, "order_number": 2},
        )
        invalid_top_k = self.client.post(
            "/customers/10/recommendations/explain",
            json={"product_id": selected["product_id"], "top_k": 21, "order_number": 2},
        )
        with patch(
            "backend.api.main.generate_ai_explanation",
            return_value="This product matches verified shopping evidence.",
        ) as explain:
            response = self.client.post(
                "/customers/10/recommendations/explain",
                json={
                    "product_id": selected["product_id"],
                    "top_k": 2,
                    "order_number": 2,
                },
            )

        self.assertEqual(invalid_product.status_code, 422)
        self.assertEqual(invalid_top_k.status_code, 422)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reason_codes"], selected["explanation_reason_codes"])
        explain.assert_called_once_with(
            selected["product_name"], selected["explanation_reason_codes"]
        )

    def test_recommendation_order_is_deterministic(self):
        first = self.client.get("/customers/10/recommendations?top_k=5").json()
        second = self.client.get("/customers/10/recommendations?top_k=5").json()
        self.assertEqual(first, second)

    def test_model_is_loaded_once_for_application_lifetime(self):
        application = create_app(history=self.history, model_path=DEFAULT_MODEL_PATH)
        with patch("backend.api.main.load_model", wraps=load_model) as model_loader:
            with TestClient(application) as client:
                self.assertEqual(client.get("/customers/10/recommendations?top_k=1").status_code, 200)
                self.assertEqual(client.get("/customers/10/recommendations?top_k=1").status_code, 200)
            model_loader.assert_called_once()

    def test_top_k_validation(self):
        self.assertEqual(self.client.get("/customers/10/recommendations?top_k=0").status_code, 422)
        self.assertEqual(self.client.get("/customers/10/recommendations?top_k=21").status_code, 422)
        self.assertEqual(self.client.get("/customers/10/recommendations?top_k=bad").status_code, 422)

    def test_invalid_and_unknown_customer_ids(self):
        self.assertEqual(self.client.get("/customers/not-an-id/recommendations").status_code, 422)
        self.assertEqual(self.client.get("/customers/0/recommendations").status_code, 422)
        response = self.client.get("/customers/999/recommendations")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Customer was not found.")

    def test_insufficient_history_is_a_clean_bad_request(self):
        response = self.client.get("/customers/20/recommendations")
        self.assertEqual(response.status_code, 400)
        self.assertIn("insufficient", response.json()["detail"])

    def test_sparse_order_number_does_not_fake_earlier_history(self):
        altered_orders = self.orders.copy()
        altered_orders.loc[altered_orders["order_id"] == 201, "order_number"] = 2
        sparse_history = HistoryStore(
            altered_orders, self.prior, self.products, self.aisles, self.departments
        )
        with TestClient(create_app(history=sparse_history, model=self.model)) as client:
            response = client.get("/customers/20/recommendations")
        self.assertEqual(response.status_code, 400)
        self.assertIn("insufficient", response.json()["detail"])

    def test_order_number_cutoff_validation(self):
        response = self.client.get("/customers/10/recommendations?top_k=2&order_number=2")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["recommendations"]), 2)
        self.assertEqual(
            self.client.get("/customers/10/recommendations?order_number=1").status_code,
            400,
        )
        self.assertEqual(
            self.client.get("/customers/10/recommendations?order_number=99").status_code,
            400,
        )
        self.assertEqual(
            self.client.get("/customers/10/recommendations?order_number=0").status_code,
            422,
        )

    def test_target_and_future_orders_do_not_affect_earlier_cutoff(self):
        original = self.client.get(
            "/customers/10/recommendations?top_k=5&order_number=2"
        ).json()
        altered_prior = self.prior.copy()
        target_rows = altered_prior["order_id"] == 102
        future_rows = altered_prior["order_id"] == 103
        altered_prior.loc[target_rows, "product_id"] = [2, 5]
        altered_prior.loc[future_rows, "product_id"] = [1, 3]
        altered_history = HistoryStore(
            self.orders, altered_prior, self.products, self.aisles, self.departments
        )
        with TestClient(create_app(history=altered_history, model=self.model)) as altered_client:
            altered = altered_client.get(
                "/customers/10/recommendations?top_k=5&order_number=2"
            ).json()
        self.assertEqual(original, altered)

    def test_customer_summary_uses_prior_history(self):
        response = self.client.get("/customers/10/summary")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "customer_id": 10,
                "total_orders": 3,
                "products_purchased": 4,
                "most_active_department_id": 100,
                "recent_order_number": 3,
            },
        )

    def test_missing_model_artifact_returns_safe_service_error(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(
                history=self.history,
                model_path=Path(directory) / "missing-model.joblib",
            )
            with TestClient(app) as client:
                response = client.get("/customers/10/recommendations")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["detail"], "The recommendation model is unavailable.")
                self.assertNotIn(directory, response.text)

    def test_missing_index_returns_safe_service_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = create_app(
                raw_dir=root / "raw",
                index_path=root / "missing-history.sqlite",
                model_path=DEFAULT_MODEL_PATH,
            )
            with TestClient(app) as client:
                response = client.get("/health")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["detail"], "Indexed customer history is unavailable.")
                self.assertNotIn(directory, response.text)

    def test_unexpected_error_does_not_expose_internal_details(self):
        application = create_app(history=self.history, model=self.model)
        with patch("backend.api.main.recommend_products", side_effect=RuntimeError("D:\\private\\trace")):
            with TestClient(application, raise_server_exceptions=False) as client:
                response = client.get("/customers/10/recommendations?top_k=1")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "An unexpected internal error occurred."})
        self.assertNotIn("private", response.text)


if __name__ == "__main__":
    unittest.main()