"""
SharePoint Document Chatbot API – LangGraph agent edition.

Graph: START -> agent <-> tools (rag_search, general_query) -> END
LLM: Groq via LiteLLM gateway (LITELLM_MODEL / GROQ_API_KEY)
"""

from __future__ import annotations

import os
import re
import asyncio
import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, StrictStr

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from agent_graph import run_agent
from guardrails import validate_request
from llm_gateway import gateway_configuration_error, set_request_id

try:
    from search_service import (
        ingest_documents,
        search_documents,
        ingest_sharepoint_documents,
    )
except (ImportError, KeyError):
    ingest_documents = None
    search_documents = None
    ingest_sharepoint_documents = None

logger = logging.getLogger(__name__)
configured_log_level = getattr(
    logging,
    os.getenv("LOG_LEVEL", "INFO").strip().upper(),
    None,
)
if isinstance(configured_log_level, int):
    logging.getLogger().setLevel(configured_log_level)
else:
    logger.warning("Invalid LOG_LEVEL setting; using existing logging configuration")

documents_path = Path(os.getenv("DOCUMENTS_DIR", "../documents"))
if not documents_path.is_absolute():
    documents_path = (BASE_DIR / documents_path).resolve()

app = FastAPI(
    title="SharePoint Document Chatbot API (LangGraph Agent)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://192.168.10.8:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: StrictStr = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: StrictStr = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    question: StrictStr = Field(min_length=1, max_length=20000)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=50)


_local_rate_counts: dict[tuple[str, int], int] = {}
_local_rate_lock = asyncio.Lock()
_redis_rate_client = None
_chat_slots = asyncio.Semaphore(max(1, int(os.getenv("API_MAX_CONCURRENT_CHATS", "10"))))
_chat_rate = max(1, int(os.getenv("API_RATE_LIMIT_REQUESTS", "20")))
_chat_rate_window = max(1, int(os.getenv("API_RATE_LIMIT_WINDOW_SECONDS", "60")))
_request_timeout = max(1, float(os.getenv("API_REQUEST_TIMEOUT_SECONDS", "180")))


class RequestSecurityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from starlette.requests import Request as StarletteRequest
        from starlette.responses import JSONResponse

        request = StarletteRequest(scope, receive=receive)
        supplied_id = request.headers.get("x-request-id", "")
        request_id = (
            supplied_id
            if len(supplied_id) <= 128 and re.fullmatch(r"[A-Za-z0-9._-]+", supplied_id)
            else str(uuid.uuid4())
        )
        scope.setdefault("state", {})["request_id"] = request_id
        set_request_id(request_id)

        if request.url.path == "/chat" and request.method == "POST":
            remote = request.client.host if request.client else "unknown"
            rate_subject = hashlib.sha256(remote.encode()).hexdigest()
            bucket = int(time.time()) // _chat_rate_window
            if _redis_rate_client is not None:
                key = f"chat-rate:{rate_subject}:{bucket}"
                try:
                    count = await _redis_rate_client.incr(key)
                    if count == 1:
                        await _redis_rate_client.expire(key, _chat_rate_window + 1)
                except Exception as exc:
                    logger.error(
                        "Distributed rate limiter unavailable",
                        extra={"request_id": request_id, "error_category": type(exc).__name__},
                    )
                    response = JSONResponse(
                        {"detail": "Rate limiting is temporarily unavailable."},
                        status_code=503,
                    )
                    response.headers["X-Request-ID"] = request_id
                    await response(scope, receive, send)
                    set_request_id(None)
                    return
            else:
                if os.getenv("APP_ENV", "development").lower() == "production":
                    response = JSONResponse(
                        {"detail": "Distributed rate limiting is not configured."},
                        status_code=503,
                    )
                    response.headers["X-Request-ID"] = request_id
                    await response(scope, receive, send)
                    set_request_id(None)
                    return
                async with _local_rate_lock:
                    stale = [key for key in _local_rate_counts if key[1] < bucket - 1]
                    for key in stale:
                        _local_rate_counts.pop(key, None)
                    rate_key = (rate_subject, bucket)
                    _local_rate_counts[rate_key] = _local_rate_counts.get(rate_key, 0) + 1
                    count = _local_rate_counts[rate_key]
            if count > _chat_rate:
                logger.warning(
                    "Chat request rate limit exceeded",
                    extra={"request_id": request_id, "outcome": "rate_limited"},
                )
                response = JSONResponse(
                    {"detail": "Too many chat requests. Please try again later."},
                    status_code=429,
                    headers={"Retry-After": str(_chat_rate_window)},
                )
                response.headers["X-Request-ID"] = request_id
                await response(scope, receive, send)
                set_request_id(None)
                return

            try:
                await asyncio.wait_for(_chat_slots.acquire(), timeout=1)
            except asyncio.TimeoutError:
                response = JSONResponse(
                    {"detail": "Chat service is at capacity. Please retry shortly."},
                    status_code=503,
                    headers={"Retry-After": "1", "X-Request-ID": request_id},
                )
                await response(scope, receive, send)
                set_request_id(None)
                return
            try:
                await self._send_with_request_id(scope, receive, send, request_id)
            finally:
                if not scope.get("state", {}).get("hold_chat_slot"):
                    _chat_slots.release()
                set_request_id(None)
            return

        await self._send_with_request_id(scope, receive, send, request_id)
        set_request_id(None)

    async def _send_with_request_id(self, scope, receive, send, request_id):
        started = time.monotonic()
        response_status = None

        async def send_with_id(message):
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            logger.info(
                "HTTP request completed",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "status_code": response_status,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                },
            )


app.add_middleware(RequestSecurityMiddleware)


@app.on_event("startup")
async def configure_rate_limiter():
    global _redis_rate_client
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            from redis.asyncio import Redis

            _redis_rate_client = Redis.from_url(redis_url, decode_responses=True)
            await _redis_rate_client.ping()
        except Exception as exc:
            _redis_rate_client = None
            logger.error(
                "Could not initialize distributed rate limiter",
                extra={"error_category": type(exc).__name__},
            )
            raise RuntimeError("REDIS_URL is configured but Redis is unavailable.") from exc
    if os.getenv("APP_ENV", "development").lower() == "production" and not redis_url:
        raise RuntimeError("REDIS_URL is required in production for distributed rate limiting.")


@app.on_event("shutdown")
async def close_rate_limiter():
    if _redis_rate_client is not None:
        await _redis_rate_client.aclose()


def _release_chat_slot_when_done(worker_task: asyncio.Task, loop) -> None:
    try:
        error = worker_task.exception()
    except asyncio.CancelledError:
        error = None
    if error is not None:
        logger.error(
            "Timed-out chat worker failed after the HTTP response",
            extra={"error_category": type(error).__name__},
        )
    loop.call_soon_threadsafe(_chat_slots.release)


@app.get("/")
def root():
    return {
        "message": "Document chatbot API (LangGraph agent) is running",
        "docs": "/docs",
        "graph": "START -> agent <-> tools -> END",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    error = gateway_configuration_error()
    if error:
        raise HTTPException(status_code=503, detail=error)
    if search_documents is None:
        raise HTTPException(status_code=503, detail="Search service is not configured.")
    return {"status": "ready"}


@app.get("/documents/{filename}")
def preview_document(filename: str):
    extension = Path(filename).suffix.lower()
    if (
        Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or extension not in {".pdf", ".pptx"}
    ):
        raise HTTPException(status_code=400, detail="Unsupported document.")

    candidates = []
    documents_root = documents_path.resolve()
    candidates.extend(
        path
        for path in documents_root.rglob(filename)
        if path.is_file() and path.resolve().is_relative_to(documents_root)
    )

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    sharepoint_cache = BASE_DIR / ".cache" / "sharepoint"
    if sharepoint_cache.is_dir():
        candidates.extend(
            path
            for path in sharepoint_cache.glob(f"*_{safe_name}")
            if path.is_file()
            and path.resolve().is_relative_to(sharepoint_cache.resolve())
        )

    if not candidates:
        raise HTTPException(status_code=404, detail="Document not found.")
    if len(candidates) > 1:
        raise HTTPException(
            status_code=409,
            detail="Multiple documents have this filename; preview is ambiguous.",
        )

    media_type = (
        "application/pdf"
        if extension == ".pdf"
        else "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
    return FileResponse(
        candidates[0],
        media_type=media_type,
        filename=filename,
        content_disposition_type="inline",
    )


@app.post("/ingest")
def ingest():
    if ingest_documents is None:
        raise HTTPException(status_code=501, detail="search_service not available")
    try:
        return ingest_documents(documents_path)
    except Exception as exc:
        logger.error("Document ingestion failed", extra={"error_category": type(exc).__name__})
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        ) from exc


@app.post("/search")
def search(request: SearchRequest):
    if search_documents is None:
        raise HTTPException(status_code=501, detail="search_service not available")
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Search query must not be empty.")
    try:
        results = search_documents(request.query, request.top_k)
        return {"query": request.query, "results": results}
    except Exception as exc:
        logger.error("Search failed", extra={"error_category": type(exc).__name__})
        raise HTTPException(
            status_code=500,
            detail="Search failed.",
        ) from exc


@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    """
    LangGraph agent endpoint.

    Flow:
      START -> agent (LiteLLM/Groq) <-> tools (rag_search | general_query)
            -> loop until agent produces final answer -> END
    """
    try:
        history = [
            HumanMessage(content=message.content)
            if message.role == "user"
            else AIMessage(content=message.content)
            for message in request.history
        ]
        question, history = validate_request(request.question, history)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    configuration_error = gateway_configuration_error()
    if configuration_error:
        raise HTTPException(
            status_code=503,
            detail=configuration_error,
        )

    try:
        worker_task = asyncio.create_task(
            asyncio.to_thread(run_agent, question, history, already_validated=True)
        )
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                asyncio.shield(worker_task),
                timeout=_request_timeout,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            http_request.state.hold_chat_slot = True
            worker_task.add_done_callback(
                lambda completed: _release_chat_slot_when_done(completed, loop)
            )
            raise
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "citations": result.get("citations", []),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        logger.error(
            "Chat request timed out",
            extra={"request_id": http_request.state.request_id},
        )
        raise HTTPException(status_code=504, detail="Chat request timed out.") from exc
    except Exception as exc:
        logger.error(
            "Chat request failed",
            extra={
                "request_id": http_request.state.request_id,
                "error_category": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=502,
            detail="Chat request failed. Please try again later.",
        ) from exc


@app.post("/sharepoint/sync")
def sync_sharepoint():
    if ingest_sharepoint_documents is None:
        raise HTTPException(status_code=501, detail="search_service not available")
    try:
        return ingest_sharepoint_documents()
    except Exception as exc:
        logger.error("SharePoint sync failed", extra={"error_category": type(exc).__name__})
        raise HTTPException(
            status_code=500,
            detail="SharePoint sync failed.",
        ) from exc
