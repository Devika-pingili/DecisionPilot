import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.ai.explanation import (
    MAX_EXPLANATION_LENGTH,
    SYSTEM_INSTRUCTIONS,
    generate_ai_explanation,
)


class AIExplanationServiceTests(unittest.TestCase):
    def test_unconfigured_provider_returns_unavailable_without_client_call(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "backend.ai.explanation.OpenAI"
        ) as client:
            self.assertIsNone(generate_ai_explanation("Soda", ["CUSTOMER_FREQUENT"]))
        client.assert_not_called()

    @patch.dict(
        os.environ,
        {"DECISIONPILOT_AI_API_KEY": "test-key", "DECISIONPILOT_AI_MODEL": "test-model"},
    )
    @patch("backend.ai.explanation.OpenAI")
    def test_provider_receives_only_product_and_allowlisted_evidence(self, client_factory):
        client = client_factory.return_value
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Frequently purchased."))]
        )

        result = generate_ai_explanation(
            "Soda",
            ["CUSTOMER_FREQUENT", "FAKE_REASON"],
        )

        self.assertEqual(result, "Frequently purchased.")
        client_factory.assert_called_once_with(api_key="test-key", timeout=8.0, max_retries=0)
        call = client.chat.completions.create.call_args.kwargs
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["max_tokens"], 100)
        self.assertEqual(call["messages"][0]["content"], SYSTEM_INSTRUCTIONS)
        self.assertIn("Product: Soda", call["messages"][1]["content"])
        self.assertIn("multiple previous orders", call["messages"][1]["content"])
        self.assertNotIn("FAKE_REASON", call["messages"][1]["content"])
        self.assertNotIn("customer_id", call["messages"][1]["content"])

    @patch.dict(os.environ, {"DECISIONPILOT_AI_API_KEY": "test-key"})
    @patch("backend.ai.explanation.OpenAI", side_effect=TimeoutError("provider timeout"))
    def test_provider_timeout_returns_unavailable(self, _client_factory):
        self.assertIsNone(generate_ai_explanation("Soda", ["CUSTOMER_RECENT"]))

    @patch.dict(os.environ, {"DECISIONPILOT_AI_API_KEY": "test-key"})
    @patch("backend.ai.explanation.OpenAI", side_effect=RuntimeError("provider failure"))
    def test_provider_failure_returns_unavailable(self, _client_factory):
        self.assertIsNone(generate_ai_explanation("Soda", ["CUSTOMER_RECENT"]))

    @patch.dict(os.environ, {"DECISIONPILOT_AI_API_KEY": "test-key"})
    @patch("backend.ai.explanation.OpenAI")
    def test_provider_output_is_trimmed_to_maximum_length(self, client_factory):
        client_factory.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="x" * (MAX_EXPLANATION_LENGTH + 20))
                )
            ]
        )

        result = generate_ai_explanation("Soda", ["GLOBAL_POPULARITY"])

        self.assertEqual(len(result), MAX_EXPLANATION_LENGTH)


if __name__ == "__main__":
    unittest.main()
