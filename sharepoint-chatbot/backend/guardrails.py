"""LangKit-backed request validation for the document agent."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from llm_gateway import get_request_id

logger = logging.getLogger(__name__)


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("Invalid integer configuration; using default", extra={"setting": name})
        return default


def max_question_length() -> int:
    return _positive_int("AGENT_MAX_INPUT_CHARS", 2000)


def injection_threshold() -> float:
    """Return the LangKit score at which a request is rejected."""
    try:
        threshold = float(os.getenv("LANGKIT_INJECTION_THRESHOLD", "0.70"))
    except ValueError:
        _reject("invalid_langkit_threshold", "LANGKIT_INJECTION_THRESHOLD must be a number from -1 to 1.")
    if not -1 <= threshold <= 1:
        _reject("invalid_langkit_threshold", "LANGKIT_INJECTION_THRESHOLD must be a number from -1 to 1.")
    return threshold


def _langkit_injection_score(prompt: str) -> float:
    """Run LangKit's injection similarity detector for one untrusted prompt."""
    try:
        # Importing this module initializes LangKit's local embedding model and
        # known-injection embedding set.
        from langkit import injections

        scores = injections.injection({"prompt": [prompt]})
        score = float(scores[0])
    except Exception as exc:
        logger.exception(
            "LangKit injection detector unavailable",
            extra={"request_id": get_request_id(), "error_category": type(exc).__name__},
        )
        _reject("langkit_unavailable", "Safety screening is temporarily unavailable. Please try again later.")
    if not -1 <= score <= 1:
        _reject("invalid_langkit_score", "Safety screening returned an invalid result.")
    return score


def validate_request(
    question: str,
    history: Sequence[BaseMessage] | None = None,
) -> tuple[str, list[BaseMessage]]:
    """Validate request shape, bound context, and reject LangKit-detected injections."""
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

    # The model receives retained history beside the current question, so screen
    # that exact untrusted context rather than only the latest turn.
    candidate = "\n".join([question, *(m.content for m in bounded_history)])
    score = _langkit_injection_score(candidate)
    threshold = injection_threshold()
    if score >= threshold:
        logger.warning(
            "Request rejected by LangKit injection guardrail",
            extra={
                "request_id": get_request_id(),
                "guardrail": "langkit_injection",
                "score": score,
                "threshold": threshold,
            },
        )
        _reject("langkit_injection", "Request was blocked by the safety policy.")

    logger.info(
        "LangKit injection screening passed",
        extra={
            "request_id": get_request_id(),
            "guardrail": "langkit_injection",
            "score": score,
            "threshold": threshold,
        },
    )
    return question.strip(), bounded_history


def _reject(reason: str, message: str) -> None:
    logger.warning(
        "Request rejected by guardrail",
        extra={"request_id": get_request_id(), "rejection_reason": reason},
    )
    raise ValueError(message)
