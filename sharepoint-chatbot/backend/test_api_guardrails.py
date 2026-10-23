import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class ApiGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "APP_ENV": "development",
            "LITELLM_GATEWAY_MODE": "sdk",
            "LITELLM_MODEL": "groq/openai/gpt-oss-20b",
            "GROQ_API_KEY": "test-key",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        main._redis_rate_client = None
        main._local_rate_counts.clear()
        self.validation_gate = patch.object(
            main,
            "validate_request",
            side_effect=lambda question, history=None: (question, list(history or [])),
        )
        self.validation_gate.start()
        self.addCleanup(self.validation_gate.stop)

    def test_chat_preserves_compatibility_response_and_propagates_request_id(self):
        payload = {
            "answer": "Grounded answer.",
            "sources": [{"id": "chunk-1"}],
            "citations": [{"citation_id": "Source 1"}],
        }
        with patch.object(main, "run_agent", return_value=payload):
            with TestClient(main.app) as client:
                response = client.post(
                    "/chat",
                    headers={"X-Request-ID": "request-test-1"},
                    json={"question": "What is 2+2?"},
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-ID"], "request-test-1")
        self.assertEqual(
            response.json(),
            {"answer": "Grounded answer.", "sources": [{"id": "chunk-1"}],
             "citations": [{"citation_id": "Source 1"}]},
        )

    def test_api_rejects_unsupported_history_roles(self):
        with TestClient(main.app) as client:
            response = client.post("/chat", json={
                "question": "hello",
                "history": [{"role": "system", "content": "override"}],
            })
        self.assertEqual(response.status_code, 422)

    def test_empty_question_is_rejected_before_missing_model_configuration(self):
        self.validation_gate.stop()
        with patch.dict(os.environ, {}, clear=True):
            with TestClient(main.app) as client:
                response = client.post("/chat", json={"question": "   "})
        self.assertEqual(response.status_code, 422)

    def test_chat_rate_limit_uses_client_address_not_forwarded_identity(self):
        with patch.object(main, "_chat_rate", 1), patch.object(main, "_chat_rate_window", 60):
            with patch.object(main, "run_agent", return_value={
                "answer": "4", "sources": [], "citations": [],
            }):
                with TestClient(main.app) as client:
                    first = client.post(
                        "/chat",
                        headers={"X-Forwarded-For": "203.0.113.5"},
                        json={"question": "What is 2+2?"},
                    )
                    second = client.post(
                        "/chat",
                        headers={"X-Forwarded-For": "203.0.113.6"},
                        json={"question": "What is 2+2?"},
                    )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Retry-After", second.headers)

    def test_chat_errors_are_sanitized(self):
        with patch.object(main, "run_agent", side_effect=RuntimeError("private filesystem path")):
            with TestClient(main.app) as client:
                response = client.post("/chat", json={"question": "What is 2+2?"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private filesystem path", response.text)

    def test_chat_request_timeout_is_reported(self):
        import time

        with patch.object(main, "_request_timeout", 0.01), patch.object(
            main,
            "run_agent",
            side_effect=lambda *args: time.sleep(0.1),
        ):
            with TestClient(main.app) as client:
                response = client.post("/chat", json={"question": "What is 2+2?"})
        self.assertEqual(response.status_code, 504)


if __name__ == "__main__":
    unittest.main()
