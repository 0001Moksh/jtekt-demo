"""
SharePoint Document Chatbot API – LangGraph agent edition.

Graph: START -> agent <-> tools (rag_search, general_query) -> END
LLM: Groq via LiteLLM gateway (LITELLM_MODEL / GROQ_API_KEY)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent_graph import run_agent

try:
    from search_service import (
        ingest_documents,
        search_documents,
        ingest_sharepoint_documents,
    )
except ImportError:
    ingest_documents = None
    search_documents = None
    ingest_sharepoint_documents = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

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
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


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
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc


@app.post("/search")
def search(request: SearchRequest):
    if search_documents is None:
        raise HTTPException(status_code=501, detail="search_service not available")
    try:
        results = search_documents(request.query, request.top_k)
        return {"query": request.query, "results": results}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}",
        ) from exc


@app.post("/chat")
def chat(request: ChatRequest):
    """
    LangGraph agent endpoint.

    Flow:
      START -> agent (LiteLLM/Groq) <-> tools (rag_search | general_query)
            -> loop until agent produces final answer -> END
    """
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY (or LITELLM_API_KEY) is not configured",
        )

    try:
        result = run_agent(request.question)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "citations": result.get("citations", []),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {exc}",
        ) from exc


@app.post("/sharepoint/sync")
def sync_sharepoint():
    if ingest_sharepoint_documents is None:
        raise HTTPException(status_code=501, detail="search_service not available")
    try:
        return ingest_sharepoint_documents()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SharePoint sync failed: {exc}",
        ) from exc