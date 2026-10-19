"""Deterministic validation and privacy controls for agent requests."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from llm_gateway import get_request_id

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.I),
    re.compile(r"\b(?:reveal|print|repeat|show)\s+(?:the\s+)?(?:system prompt|hidden instructions)\b", re.I),
    re.compile(r"\b(?:jailbreak|developer mode|DAN mode)\b", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
)
PII_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._~+/-]+=*)\b", re.I),
)


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("Invalid integer configuration; using default", extra={"setting": name})
        return default


def max_question_length() -> int:
    return _positive_int("AGENT_MAX_INPUT_CHARS", 2000)


def validate_request(
    question: str,
    history: Sequence[BaseMessage] | None = None,
) -> tuple[str, list[BaseMessage]]:
    """Validate request shape and bound history before constructing the graph."""
    if not isinstance(question, str) or not question.strip():
        _reject("empty_question", "Question must not be empty.")
    if len(question) > max_question_length():
        _reject("input_too_long", f"Question exceeds the {max_question_length()} character limit.")

    if history is None:
        history = ()
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        _reject("invalid_history_shape", "Conversation history must be a list of messages.")

    history_limit = _positive_int("AGENT_MAX_HISTORY_MESSAGES", 12)
    token_limit = _positive_int("AGENT_MAX_CONTEXT_TOKENS", 6000)
    bounded_history = list(history[-history_limit:])
    for message in bounded_history:
        if not isinstance(message, (HumanMessage, AIMessage)):
            _reject("invalid_history_role", "Conversation history supports only user and assistant text messages.")
        if isinstance(message, AIMessage) and message.tool_calls:
            _reject("history_tool_calls", "Conversation history cannot contain pre-existing tool calls.")
        if not isinstance(message.content, str) or not message.content.strip():
            _reject("empty_history_message", "Conversation messages must contain non-empty text.")

    estimated_tokens = (len(question) + sum(len(m.content) for m in bounded_history) + 3) // 4
    if estimated_tokens > token_limit:
        _reject("context_limit", "Question and conversation history exceed the context limit.")

    combined = "\n".join([question, *(m.content for m in bounded_history)])
    if any(pattern.search(combined) for pattern in INJECTION_PATTERNS):
        logger.warning(
            "Prompt-injection risk signal detected; request allowed with untrusted-input handling",
            extra={"request_id": get_request_id(), "guardrail": "prompt_injection_risk"},
        )

    privacy_mode = os.getenv("PII_MODE", "off").strip().lower()
    if privacy_mode not in {"off", "redact", "reject"}:
        _reject("invalid_pii_mode", "PII_MODE must be off, redact, or reject.")
    if privacy_mode != "off":
        detected = any(pattern.search(combined) for pattern in PII_PATTERNS)
        if detected and privacy_mode == "reject":
            _reject("pii_rejected", "Request contains a sensitive pattern that is not accepted.")
        if detected:
            question = _redact(question)
            bounded_history = [
                type(message)(content=_redact(message.content))
                for message in bounded_history
            ]
            logger.info(
                "Sensitive patterns redacted from request",
                extra={"request_id": get_request_id(), "guardrail": "pii_redacted"},
            )

    return question.strip(), bounded_history


def _redact(text: str) -> str:
    for pattern in PII_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def redact_sensitive_text(text: str) -> str:
    """Redact the supported PII/credential patterns from generated output."""
    return _redact(text)


def _reject(reason: str, message: str) -> None:
    logger.warning(
        "Request rejected by guardrail",
        extra={"request_id": get_request_id(), "rejection_reason": reason},
    )
    raise ValueError(message)
