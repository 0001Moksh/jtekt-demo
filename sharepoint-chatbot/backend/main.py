from search_service import (
    ingest_documents,
    search_documents,
    ingest_sharepoint_documents,
)
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from groq import Groq

from search_service import ingest_documents, search_documents
from rag_utils import SYSTEM_PROMPT, build_context, expand_source_citations

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

documents_path = Path(
    os.getenv("DOCUMENTS_DIR", "../documents")
)
if not documents_path.is_absolute():
    documents_path = (BASE_DIR / documents_path).resolve()

app = FastAPI(
    title="SharePoint Document Chatbot API",
    version="1.0.0",
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
        "message": "Document chatbot API is running",
        "docs": "/docs",
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
        path for path in documents_root.rglob(filename)
        if path.is_file() and path.resolve().is_relative_to(documents_root)
    )

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    sharepoint_cache = BASE_DIR / ".cache" / "sharepoint"
    if sharepoint_cache.is_dir():
        candidates.extend(
            path for path in sharepoint_cache.glob(f"*_{safe_name}")
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
    try:
        return ingest_documents(documents_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc


@app.post("/search")
def search(request: SearchRequest):
    try:
        results = search_documents(
            request.query, request.top_k
        )
        return {"query": request.query, "results": results}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}",
        ) from exc


@app.post("/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured",
        )

    try:
        sources = search_documents(request.question, top_k=10)

        if not sources:
            return {
                "answer": "I could not find relevant information in the indexed documents.",
                "sources": [],
            }

        context = build_context(sources)

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{request.question}\n\n"
                        f"Document context:\n{context}"
                    ),
                },
            ],
        )
        answer = response.choices[0].message.content or ""
        try:
            answer, citations = expand_source_citations(answer, sources)
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail=str(exc),
            ) from exc

        return {
            "answer": answer,
            "sources": sources,
            "citations": citations,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {exc}",
        ) from exc


@app.post("/sharepoint/sync")
def sync_sharepoint():
    try:
        return ingest_sharepoint_documents()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SharePoint sync failed: {exc}",
        ) from exc
