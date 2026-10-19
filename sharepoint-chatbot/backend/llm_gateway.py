"""Central LiteLLM SDK/proxy gateway with bounded retries and safe telemetry."""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit

try:
    from litellm import completion as _litellm_completion
except ImportError:
    _litellm_completion = None

try:
    from litellm import supports_function_calling as _supports_function_calling
except ImportError:
    _supports_function_calling = None

try:
    from openai import Timeout as OpenAITimeout
except ImportError:
    OpenAITimeout = None

logger = logging.getLogger(__name__)
_request_id: ContextVar[str] = ContextVar("request_id", default="unknown")


def _initial_concurrency() -> int:
    try:
        return max(1, int(os.getenv("LITELLM_MAX_CONCURRENT_REQUESTS", "10")))
    except ValueError:
        logger.warning("Invalid LITELLM_MAX_CONCURRENT_REQUESTS; using default")
        return 10


_slot_capacity = _initial_concurrency()
_slots = threading.BoundedSemaphore(_slot_capacity)
_capability_cache: dict[str, bool | None] = {}


def set_request_id(request_id: str | None) -> None:
    _request_id.set(request_id or "unknown")


def get_request_id() -> str:
    return _request_id.get()


def _int_setting(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("Invalid integer configuration; using default", extra={"setting": name})
        return default


def _bool_setting(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _category(error: Exception) -> str:
    name = type(error).__name__.casefold()
    status = getattr(error, "status_code", None)
    if status in {401, 403} or any(word in name for word in ("auth", "permission")):
        return "authentication"
    if status == 429 or "ratelimit" in name or "quota" in name:
        return "rate_limit_or_quota"
    if status is not None and 400 <= status < 500:
        return "invalid_request"
    if isinstance(error, (TimeoutError,)) or "timeout" in name:
        return "timeout"
    if any(word in name for word in ("connection", "serviceunavailable", "internalserver")):
        return "transient"
    if status is not None and status >= 500:
        return "transient"
    return "unknown"


def _retryable(category: str) -> bool:
    return category in {"rate_limit_or_quota", "timeout", "transient"}


def _models() -> tuple[str, list[str]]:
    primary = os.getenv("LITELLM_MODEL", "groq/openai/gpt-oss-20b").strip()
    route_pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")
    if not route_pattern.fullmatch(primary):
        raise ValueError("LITELLM_MODEL must be a non-empty model route.")
    fallbacks = [
        model.strip()
        for model in os.getenv("LITELLM_FALLBACK_MODELS", "").split(",")
        if model.strip()
    ]
    if any(not route_pattern.fullmatch(model) for model in fallbacks):
        raise ValueError("LITELLM_FALLBACK_MODELS contains an invalid model route.")
    fallbacks = list(dict.fromkeys(model for model in fallbacks if model != primary))
    if len(fallbacks) > 3:
        raise ValueError("LITELLM_FALLBACK_MODELS supports at most three routes.")
    return primary, fallbacks


def gateway_configuration_error() -> str | None:
    """Return a safe configuration error without exposing credentials."""
    if _litellm_completion is None:
        return "LiteLLM is not installed."
    mode = os.getenv("LITELLM_GATEWAY_MODE", "sdk").strip().lower()
    if mode == "proxy":
        proxy_url = os.getenv("LITELLM_PROXY_URL", "").strip()
        parsed_url = urlsplit(proxy_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            return "LITELLM_PROXY_URL is required in proxy mode."
    elif mode != "sdk":
        return "LITELLM_GATEWAY_MODE must be sdk or proxy."
    try:
        model, fallbacks = _models()
    except ValueError as exc:
        return str(exc)
    if mode == "sdk":
        provider = model.split("/", 1)[0].upper().replace("-", "_")
        key = os.getenv(f"{provider}_API_KEY") or os.getenv("LITELLM_API_KEY")
        if provider.lower() in {"bedrock", "bedrock_converse"}:
            key = key or (
                os.getenv("AWS_ACCESS_KEY_ID")
                and os.getenv("AWS_SECRET_ACCESS_KEY")
            )
        elif provider.lower() in {"vertex_ai", "vertex_ai_beta"}:
            key = key or os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("VERTEXAI_PROJECT")
        elif provider.lower() in {"ollama", "ollama_chat", "vllm", "lm_studio"}:
            key = key or "local-provider"
        if not key:
            return f"Configure {provider}_API_KEY or LITELLM_API_KEY for the selected model."
    try:
        if float(os.getenv("LITELLM_TIMEOUT", "30")) <= 0:
            return "LITELLM_TIMEOUT must be greater than zero."
        if float(os.getenv("LITELLM_CONNECT_TIMEOUT", "10")) <= 0:
            return "LITELLM_CONNECT_TIMEOUT must be greater than zero."
    except ValueError:
        return "LiteLLM timeout settings must be numeric."
    capability = _tool_call_capability(model)
    if capability is False:
        return "The configured LITELLM_MODEL does not support tool calling."
    if any(_tool_call_capability(fallback) is False for fallback in fallbacks):
        return "A configured LITELLM_FALLBACK_MODELS route does not support tool calling."
    return None


def _tool_call_capability(model: str) -> bool | None:
    if model in _capability_cache:
        return _capability_cache[model]
    if _supports_function_calling is None:
        logger.warning(
            "LiteLLM tool-call capability checker is unavailable",
            extra={"model": model},
        )
        return None
    try:
        capability = bool(_supports_function_calling(model=model))
    except Exception as exc:
        logger.warning(
            "Could not verify model tool-call capability",
            extra={"model": model, "error_category": type(exc).__name__},
        )
        capability = None
    _capability_cache[model] = capability
    return capability


def _request_options(model: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    options = dict(kwargs)
    mode = os.getenv("LITELLM_GATEWAY_MODE", "sdk").strip().lower()
    if mode not in {"sdk", "proxy"}:
        raise ValueError("LITELLM_GATEWAY_MODE must be sdk or proxy.")
    if mode == "proxy":
        proxy_url = os.getenv("LITELLM_PROXY_URL", "").strip()
        parsed_url = urlsplit(proxy_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("LITELLM_PROXY_URL is required in proxy mode.")
        options["api_base"] = proxy_url
        proxy_key = os.getenv("LITELLM_PROXY_API_KEY") or os.getenv("LITELLM_API_KEY")
        if proxy_key:
            options["api_key"] = proxy_key
        else:
            options.pop("api_key", None)
    else:
        options.pop("api_base", None)
        options.pop("api_key", None)
        provider = model.split("/", 1)[0].upper().replace("-", "_")
        provider_key = os.getenv(f"{provider}_API_KEY")
        shared_key = os.getenv("LITELLM_API_KEY")
        if provider_key or shared_key:
            options["api_key"] = provider_key or shared_key

    timeout = max(0.1, float(os.getenv("LITELLM_TIMEOUT", "30")))
    connect_timeout = max(0.1, float(os.getenv("LITELLM_CONNECT_TIMEOUT", "10")))
    options["timeout"] = (
        OpenAITimeout(
            connect=connect_timeout,
            read=timeout,
            write=timeout,
            pool=connect_timeout,
        )
        if OpenAITimeout is not None
        else timeout
    )
    options["model"] = model
    output_limit = _int_setting("LITELLM_MAX_OUTPUT_TOKENS", 2048)
    requested_tokens = options.pop("max_completion_tokens", None)
    if requested_tokens is not None:
        options["max_tokens"] = min(int(requested_tokens), output_limit)
    elif "max_tokens" in options:
        options["max_tokens"] = min(int(options["max_tokens"]), output_limit)
    else:
        options["max_tokens"] = output_limit
    options["num_retries"] = 0
    return options


def _log_usage(response: Any, model: str) -> None:
    usage = getattr(response, "usage", None)
    fields = {
        "request_id": get_request_id(),
        "model": model,
        "provider": model.split("/", 1)[0],
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    if _bool_setting("LITELLM_ENABLE_COST_TRACKING"):
        hidden = getattr(response, "_hidden_params", {}) or {}
        fields["estimated_cost"] = hidden.get("response_cost")
    logger.info("LLM usage", extra=fields)


def completion(**kwargs: Any) -> Any:
    """Route a LiteLLM completion through shared limits, retry, and telemetry."""
    if _litellm_completion is None:
        raise RuntimeError("LiteLLM is not installed. Install backend/requirements.txt.")
    primary, fallbacks = _models()
    configured_model = kwargs.pop("model", primary)
    if configured_model != primary:
        # Keep explicit model selection only when it is the configured primary.
        raise ValueError("LLM calls must use the centrally configured LITELLM_MODEL.")
    candidates = [primary, *fallbacks]
    timeout = max(0.1, float(os.getenv("LITELLM_TIMEOUT", "30")))
    capacity = _int_setting("LITELLM_MAX_CONCURRENT_REQUESTS", 10)
    if capacity != _slot_capacity:
        logger.warning(
            "Concurrency setting changed after gateway initialization; restart to apply"
        )
    if not _slots.acquire(timeout=timeout):
        raise TimeoutError("LLM concurrency limit timed out.")
    try:
        max_retries = _int_setting("LITELLM_MAX_RETRIES", 2, minimum=0)
        for model_index, model in enumerate(candidates):
            for attempt in range(max_retries + 1):
                started = time.monotonic()
                try:
                    if kwargs.get("tools") and _tool_call_capability(model) is False:
                        raise ValueError(f"Configured route {model} does not support tool calling.")
                    response = _litellm_completion(**_request_options(model, kwargs))
                    logger.info(
                        "LLM request succeeded",
                        extra={
                            "request_id": get_request_id(),
                            "model": model,
                            "provider": model.split("/", 1)[0],
                            "attempt": attempt + 1,
                            "latency_ms": round((time.monotonic() - started) * 1000),
                            "fallback": model_index > 0,
                        },
                    )
                    _log_usage(response, model)
                    return response
                except Exception as exc:
                    category = _category(exc)
                    logger.warning(
                        "LLM request failed",
                        extra={
                            "request_id": get_request_id(),
                            "model": model,
                            "provider": model.split("/", 1)[0],
                            "attempt": attempt + 1,
                            "latency_ms": round((time.monotonic() - started) * 1000),
                            "error_category": category,
                        },
                    )
                    if not _retryable(category):
                        raise
                    if attempt < max_retries:
                        delay = min(4.0, 0.25 * (2 ** attempt))
                        time.sleep(delay + random.uniform(0, delay * 0.25))
                    elif model_index == len(candidates) - 1:
                        raise
                    else:
                        logger.warning(
                            "LLM fallback selected",
                            extra={
                                "request_id": get_request_id(),
                                "model": model,
                                "provider": model.split("/", 1)[0],
                                "fallback_model": candidates[model_index + 1],
                                "reason": category,
                            },
                        )
        raise RuntimeError("No LLM model was available.")
    finally:
        _slots.release()
