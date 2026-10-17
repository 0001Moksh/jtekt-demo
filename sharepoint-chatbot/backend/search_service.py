import os
from pathlib import Path
import hashlib

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)
from azure.search.documents.models import VectorizedQuery
from sentence_transformers import SentenceTransformer

from document_processor import (
    SUPPORTED_EXTENSIONS,
    extract_text,
    chunk_page,
    stable_document_id,
)
from rag_utils import rerank_search_results

load_dotenv(Path(__file__).resolve().parent / ".env")

ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME = os.getenv(
    "AZURE_SEARCH_INDEX", "sharepoint-documents"
)
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
SEMANTIC_CONFIGURATION = os.getenv(
    "AZURE_SEARCH_SEMANTIC_CONFIGURATION"
)

credential = AzureKeyCredential(API_KEY)

index_client = SearchIndexClient(
    endpoint=ENDPOINT,
    credential=credential,
)

search_client = SearchClient(
    endpoint=ENDPOINT,
    index_name=INDEX_NAME,
    credential=credential,
)

# Loads the embedding model on first import.
embedding_model = SentenceTransformer(MODEL_NAME)
VECTOR_DIMENSIONS = embedding_model.get_sentence_embedding_dimension()


def create_index() -> None:
    """Create the index if it does not already exist."""
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="source_url",
            type=SearchFieldDataType.String,
        ),
        SimpleField(
            name="location",
            type=SearchFieldDataType.String,
        ),
        SearchableField(
            name="section",
            type=SearchFieldDataType.String,
        ),
        SimpleField(
            name="source_type",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="chunk_id",
            type=SearchFieldDataType.String,
        ),
        SimpleField(
            name="document_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw")
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw",
            )
        ],
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=(
            SemanticSearch(
                configurations=[_semantic_configuration()]
            )
            if SEMANTIC_CONFIGURATION
            else None
        ),
    )

    existing_names = list(index_client.list_index_names())
    if INDEX_NAME not in existing_names:
        index_client.create_index(index)
        print(f"Created Azure AI Search index: {INDEX_NAME}")
    else:
        existing_index = index_client.get_index(INDEX_NAME)
        fields_by_name = {
            field.name: field for field in existing_index.fields
        }
        vector_field = fields_by_name.get("content_vector")
        if (
            vector_field is not None
            and vector_field.vector_search_dimensions
            != VECTOR_DIMENSIONS
        ):
            raise RuntimeError(
                f"Azure AI Search index '{INDEX_NAME}' uses "
                f"{vector_field.vector_search_dimensions}-dimension vectors, "
                f"but '{MODEL_NAME}' produces {VECTOR_DIMENSIONS}. "
                "Configure a new index name and re-ingest all documents."
            )

        missing_fields = [
            field
            for field in fields
            if field.name not in fields_by_name
        ]
        index_changed = bool(missing_fields)
        if missing_fields:
            existing_index.fields.extend(missing_fields)

        if SEMANTIC_CONFIGURATION:
            semantic_search = existing_index.semantic_search
            configurations = (
                list(semantic_search.configurations or [])
                if semantic_search
                else []
            )
            if not any(
                config.name == SEMANTIC_CONFIGURATION
                for config in configurations
            ):
                configurations.append(_semantic_configuration())
                existing_index.semantic_search = SemanticSearch(
                    configurations=configurations
                )
                index_changed = True

        if index_changed:
            index_client.create_or_update_index(existing_index)
        print(f"Using existing index: {INDEX_NAME}")


def _semantic_configuration() -> SemanticConfiguration:
    if not SEMANTIC_CONFIGURATION:
        raise RuntimeError(
            "AZURE_SEARCH_SEMANTIC_CONFIGURATION is not configured."
        )
    return SemanticConfiguration(
        name=SEMANTIC_CONFIGURATION,
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[
                SemanticField(field_name="section"),
                SemanticField(field_name="content"),
            ]
        ),
    )


def _documents(
    source_type: str | None = None,
) -> list[dict[str, str]]:
    results = search_client.search(
        search_text="*",
        filter=(
            f"source_type eq '{source_type}'"
            if source_type is not None
            else None
        ),
        select=["id", "document_id"],
        top=1000,
    )
    return [
        {"id": row["id"], "document_id": row["document_id"]}
        for page in results.by_page()
        for row in page
    ]


def _legacy_sharepoint_documents() -> list[dict[str, str]]:
    results = search_client.search(
        search_text="*",
        filter="source_type eq null",
        select=["id", "document_id", "source_url"],
        top=1000,
    )
    return [
        {
            "id": row["id"],
            "document_id": row["document_id"],
            "source_url": row.get("source_url", ""),
        }
        for page in results.by_page()
        for row in page
    ]


def _document_ids(document_id: str) -> list[str]:
    results = search_client.search(
        search_text="*",
        filter=f"document_id eq '{document_id}'",
        select=["id"],
        top=1000,
    )
    return [
        row["id"]
        for page in results.by_page()
        for row in page
    ]


def _delete_ids(document_ids: list[str]) -> None:
    for start in range(0, len(document_ids), 100):
        results = search_client.delete_documents(
            documents=[
                {"id": item_id}
                for item_id in document_ids[start:start + 100]
            ]
        )
        failures = [
            result for result in results
            if not result.succeeded
        ]
        if failures:
            raise RuntimeError(
                "Azure AI Search could not delete stale chunks: "
                + "; ".join(
                    f"{result.key}: {result.error_message}"
                    for result in failures
                )
            )


def _replace_document_chunks(
    document_id: str,
    chunks: list[dict],
) -> tuple[int, list[str]]:
    old_ids = set(_document_ids(document_id=document_id))
    new_ids = {chunk["id"] for chunk in chunks}
    uploaded = 0
    errors = []

    for start in range(0, len(chunks), 100):
        results = search_client.upload_documents(
            documents=chunks[start:start + 100]
        )
        for result in results:
            if result.succeeded:
                uploaded += 1
            else:
                errors.append(
                    f"{result.key}: {result.error_message}"
                )

    if not errors:
        _delete_ids(list(old_ids - new_ids))
    return uploaded, errors


def _build_chunks(
    document_id: str,
    source_file: str,
    source_url: str,
    source_type: str,
    pages: list[dict],
) -> list[dict]:
    chunks_to_upload = []
    current_section = ""
    for page in pages:
        page_chunks = chunk_page(
            page["text"],
            initial_section=current_section,
        )
        if page_chunks:
            current_section = page_chunks[-1]["section"] or current_section

        for page_chunk in page_chunks:
            text = page_chunk["content"]
            if not text.strip():
                continue

            chunk_number = len(chunks_to_upload)
            chunk_id = f"{document_id}:{page['location']}:{chunk_number}"
            stable_chunk_id = hashlib.sha256(
                chunk_id.encode("utf-8")
            ).hexdigest()
            vector = embedding_model.encode(
                text, normalize_embeddings=True
            ).tolist()

            chunks_to_upload.append({
                "id": stable_chunk_id,
                "content": text,
                "source_file": source_file,
                "source_url": source_url,
                "location": page["location"],
                "section": page_chunk["section"],
                "chunk_id": str(chunk_number),
                "document_id": document_id,
                "source_type": source_type,
                "content_vector": vector,
            })
    return chunks_to_upload


def ingest_sharepoint_documents() -> dict:
    """Download JTEKT files from SharePoint and index their content."""
    from sharepoint_service import download_jtekt_documents

    files = download_jtekt_documents()
    create_index()

    uploaded = 0
    failures = []
    current_document_ids = set()

    for item in files:
        try:
            document_id = hashlib.sha256(
                item["item_id"].encode("utf-8")
            ).hexdigest()
            current_document_ids.add(document_id)
            chunks_to_upload = _build_chunks(
                document_id=document_id,
                source_file=item["name"],
                source_url=item["source_url"],
                source_type="sharepoint",
                pages=extract_text(item["path"]),
            )

            if not chunks_to_upload:
                failures.append({
                    "file": item["name"],
                    "error": "No extractable text found",
                })
                continue

            file_uploaded, errors = _replace_document_chunks(
                document_id,
                chunks_to_upload,
            )
            uploaded += file_uploaded
            failures.extend({
                "file": item["name"],
                "error": error,
            } for error in errors)

        except Exception as exc:
            failures.append({
                "file": item["name"],
                "error": str(exc),
            })

    if not failures:
        try:
            stale_ids = [
                row["id"]
                for row in _documents(source_type="sharepoint")
                if row["document_id"] not in current_document_ids
            ]
            stale_ids.extend(
                row["id"]
                for row in _legacy_sharepoint_documents()
                if row["source_url"]
                and row["document_id"] not in current_document_ids
            )
            _delete_ids(stale_ids)
        except Exception as exc:
            failures.append({
                "file": "SharePoint stale-document cleanup",
                "error": str(exc),
            })

    return {
        "files_found": len(files),
        "chunks_uploaded": uploaded,
        "failures": failures,
    }

def ingest_documents(documents_dir: Path) -> dict:
    """Extract, chunk, embed and upload local PDF/PPTX files."""
    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {documents_dir}"
        )

    files = [
        path for path in documents_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise ValueError(
            f"No PDF or PPTX files found in {documents_dir}"
        )

    uploaded = 0
    failures = []
    current_document_ids = set()
    legacy_document_ids = {
        hashlib.sha256(file_path.name.encode("utf-8")).hexdigest()
        for file_path in files
    }
    create_index()

    for file_path in files:
        try:
            document_id = stable_document_id(file_path, documents_dir)
            current_document_ids.add(document_id)
            chunks_to_upload = _build_chunks(
                document_id=document_id,
                source_file=file_path.name,
                source_url="",
                source_type="local",
                pages=extract_text(file_path),
            )
            if not chunks_to_upload:
                failures.append({
                    "file": file_path.name,
                    "error": "No extractable text found",
                })
                continue

            file_uploaded, errors = _replace_document_chunks(
                document_id,
                chunks_to_upload,
            )
            uploaded += file_uploaded
            failures.extend({
                "file": file_path.name,
                "error": error,
            } for error in errors)

        except Exception as exc:
            failures.append({
                "file": file_path.name,
                "error": str(exc),
            })

    if not failures:
        try:
            _delete_ids([
                item_id
                for legacy_id in legacy_document_ids
                if legacy_id not in current_document_ids
                for item_id in _document_ids(legacy_id)
            ])
            indexed_local_ids = _documents(source_type="local")
            _delete_ids([
                row["id"]
                for row in indexed_local_ids
                if row["document_id"] not in current_document_ids
            ])
        except Exception as exc:
            failures.append({
                "file": "Local stale-document cleanup",
                "error": str(exc),
            })

    return {
        "files_found": len(files),
        "chunks_uploaded": uploaded,
        "failures": failures,
    }


def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant chunks with Azure AI Search hybrid ranking."""
    if not query.strip():
        raise ValueError("Query cannot be empty")

    candidate_count = min(50, max(20, top_k * 5))
    vector = embedding_model.encode(
        query, normalize_embeddings=True
    ).tolist()

    vector_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=candidate_count,
        fields="content_vector",
    )

    search_options = {}
    if SEMANTIC_CONFIGURATION:
        search_options = {
            "query_type": "semantic",
            "semantic_configuration_name": SEMANTIC_CONFIGURATION,
        }

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=[
            "id",
            "content",
            "source_file",
            "source_url",
            "location",
            "section",
            "chunk_id",
            "document_id",
        ],
        top=candidate_count,
        **search_options,
    )

    candidates = [
        {
            "id": result["id"],
            "content": result["content"],
            "source_file": result["source_file"],
            "source_url": result.get("source_url", ""),
            "location": result["location"],
            "section": result.get("section", ""),
            "chunk_id": result["chunk_id"],
            "document_id": result.get("document_id", ""),
            "score": result.get("@search.score"),
            "reranker_score": result.get("@search.reranker_score"),
        }
        for result in results
    ]
    return rerank_search_results(query, candidates, top_k)