import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llm_gateway


def response():
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        _hidden_params={"response_cost": 0.001},
    )


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "LITELLM_GATEWAY_MODE": "sdk",
            "LITELLM_MODEL": "groq/openai/gpt-oss-20b",
            "GROQ_API_KEY": "unit-test-key",
            "LITELLM_TIMEOUT": "3",
            "LITELLM_MAX_RETRIES": "1",
            "LITELLM_MAX_OUTPUT_TOKENS": "40",
            "LITELLM_ENABLE_COST_TRACKING": "true",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_retries_transient_timeout_and_clamps_output(self):
        with patch.object(
            llm_gateway,
            "_litellm_completion",
            side_effect=[TimeoutError("private timeout detail"), response()],
        ) as call, patch.object(llm_gateway.time, "sleep"):
            result = llm_gateway.completion(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1000,
            )
        self.assertIsNotNone(result)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args.kwargs["max_tokens"], 40)
        self.assertEqual(call.call_args.kwargs["timeout"], 3)
        self.assertEqual(call.call_args.kwargs["num_retries"], 0)

    def test_rate_limit_is_retried(self):
        class RateLimitError(Exception):
            status_code = 429

        with patch.object(
            llm_gateway,
            "_litellm_completion",
            side_effect=[RateLimitError("quota detail"), response()],
        ) as call, patch.object(llm_gateway.time, "sleep"):
            llm_gateway.completion(messages=[])
        self.assertEqual(call.call_count, 2)

    def test_authentication_error_is_not_retried_or_fallen_back(self):
        class AuthenticationError(Exception):
            status_code = 401

        with patch.dict(os.environ, {"LITELLM_FALLBACK_MODELS": "groq/fallback"}):
            with patch.object(
                llm_gateway,
                "_litellm_completion",
                side_effect=AuthenticationError("secret detail"),
            ) as call:
                with self.assertRaises(AuthenticationError):
                    llm_gateway.completion(messages=[])
        self.assertEqual(call.call_count, 1)

    def test_fallback_uses_configured_route_after_retryable_failure(self):
        with patch.dict(os.environ, {
            "LITELLM_FALLBACK_MODELS": "groq/fallback-model",
            "LITELLM_MAX_RETRIES": "0",
        }):
            with patch.object(
                llm_gateway,
                "_litellm_completion",
                side_effect=[TimeoutError(), response()],
            ) as call:
                llm_gateway.completion(messages=[])
        self.assertEqual(
            [item.kwargs["model"] for item in call.call_args_list],
            ["groq/openai/gpt-oss-20b", "groq/fallback-model"],
        )

    def test_proxy_configuration_cannot_fall_through_to_sdk(self):
        with patch.dict(os.environ, {
            "LITELLM_GATEWAY_MODE": "proxy",
            "LITELLM_PROXY_URL": "https://proxy.example.test",
            "LITELLM_PROXY_API_KEY": "proxy-test-key",
        }):
            options = llm_gateway._request_options("groq/model", {"messages": []})
        self.assertEqual(options["api_base"], "https://proxy.example.test")
        self.assertEqual(options["api_key"], "proxy-test-key")

    def test_missing_credentials_has_safe_configuration_error(self):
        with patch.dict(os.environ, {
            "LITELLM_GATEWAY_MODE": "sdk",
            "LITELLM_MODEL": "groq/model",
        }, clear=True):
            self.assertEqual(
                llm_gateway.gateway_configuration_error(),
                "Configure GROQ_API_KEY or LITELLM_API_KEY for the selected model.",
            )

    def test_logs_do_not_include_provider_error_text_or_credentials(self):
        with patch.object(
            llm_gateway,
            "_litellm_completion",
            side_effect=TimeoutError("private-value-not-for-logs"),
        ), patch.object(llm_gateway.time, "sleep"):
            with self.assertLogs("llm_gateway", level="WARNING") as logs:
                with self.assertRaises(TimeoutError):
                    llm_gateway.completion(messages=[])
        self.assertNotIn("private-value-not-for-logs", "\n".join(logs.output))
        self.assertNotIn("unit-test-key", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
