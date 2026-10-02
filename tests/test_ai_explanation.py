import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.ai.explanation import (
    MAX_EXPLANATION_LENGTH,
    SYSTEM_INSTRUCTIONS,
    generate_ai_explanation,
    get_ai_provider_diagnostics,
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
        provider_input = call["messages"][1]["content"]
        self.assertEqual(
            provider_input,
            "Product: Soda\nVerified evidence:\n- The customer purchased this product in multiple previous orders.",
        )
        self.assertNotIn("FAKE_REASON", provider_input)
        self.assertNotIn("customer_id", provider_input)
        self.assertNotIn("order_id", provider_input)
        self.assertNotIn("history", provider_input.lower())

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
    def test_malformed_provider_response_returns_unavailable(self, client_factory):
        client_factory.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[]
        )

        self.assertIsNone(generate_ai_explanation("Soda", ["CUSTOMER_RECENT"]))

    @patch.dict(os.environ, {"DECISIONPILOT_AI_API_KEY": "test-key"})
    @patch("backend.ai.explanation.OpenAI")
    def test_empty_provider_response_returns_unavailable(self, client_factory):
        client_factory.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  \n "))]
        )

        self.assertIsNone(generate_ai_explanation("Soda", ["CUSTOMER_RECENT"]))

    @patch.dict(
        os.environ,
        {"DECISIONPILOT_AI_API_KEY": "test-key", "DECISIONPILOT_AI_MODEL": "   "},
    )
    @patch("backend.ai.explanation.OpenAI")
    def test_blank_model_configuration_uses_default(self, client_factory):
        client_factory.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Soda fits prior purchases."))]
        )

        generate_ai_explanation("Soda", ["CUSTOMER_FREQUENT"])

        self.assertEqual(
            client_factory.return_value.chat.completions.create.call_args.kwargs["model"],
            "gpt-4o-mini",
        )

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

    def test_diagnostics_report_missing_key_without_initializing_client(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "backend.ai.explanation.OpenAI"
        ) as client_factory:
            generate_ai_explanation("Soda", ["CUSTOMER_FREQUENT"])
            diagnostics = get_ai_provider_diagnostics()

        self.assertFalse(diagnostics["api_key_configured"])
        self.assertFalse(diagnostics["model_configured"])
        self.assertEqual(diagnostics["model_configuration"], "default")
        self.assertEqual(diagnostics["client_initialization"], "not_configured")
        self.assertFalse(diagnostics["provider_request_attempted"])
        client_factory.assert_not_called()

    @patch.dict(
        os.environ,
        {"DECISIONPILOT_AI_API_KEY": "diagnostic-test-secret", "DECISIONPILOT_AI_MODEL": "test-model"},
    )
    @patch("backend.ai.explanation.OpenAI")
    def test_diagnostics_redact_provider_exception_and_report_attempt(self, client_factory):
        client_factory.return_value.chat.completions.create.side_effect = RuntimeError(
            "Authentication failed for key=diagnostic-test-secret"
        )

        self.assertIsNone(generate_ai_explanation("Soda", ["CUSTOMER_FREQUENT"]))
        diagnostics = get_ai_provider_diagnostics()

        self.assertTrue(diagnostics["api_key_configured"])
        self.assertTrue(diagnostics["model_configured"])
        self.assertEqual(diagnostics["model_configuration"], "configured")
        self.assertEqual(diagnostics["client_initialization"], "succeeded")
        self.assertTrue(diagnostics["provider_request_attempted"])
        self.assertEqual(diagnostics["last_error_type"], "RuntimeError")
        self.assertNotIn("diagnostic-test-secret", diagnostics["last_error_message"])
        self.assertIn("[REDACTED]", diagnostics["last_error_message"])


if __name__ == "__main__":
    unittest.main()
