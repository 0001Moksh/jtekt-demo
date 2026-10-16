import os
import uuid
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
)
from azure.search.documents.models import VectorizedQuery
from sentence_transformers import SentenceTransformer

from document_processor import (
    SUPPORTED_EXTENSIONS,
    extract_text,
    chunk_text,
    stable_document_id,
)

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
    )

    existing = list(index_client.list_index_names())
    if INDEX_NAME not in existing:
        index_client.create_index(index)
        print(f"Created Azure AI Search index: {INDEX_NAME}")
    else:
        print(f"Using existing index: {INDEX_NAME}")



def ingest_sharepoint_documents() -> dict:
    """Download JTEKT files from SharePoint and index their content."""
    from sharepoint_service import download_jtekt_documents

    files = download_jtekt_documents()
    create_index()

    uploaded = 0
    failures = []

    for item in files:
        try:
            document_id = hashlib.sha256(
                item["item_id"].encode("utf-8")
            ).hexdigest()

            chunks_to_upload = []

            for page in extract_text(item["path"]):
                chunks = chunk_text(page["text"])

                for index, text in enumerate(chunks):
                    if not text.strip():
                        continue

                    chunk_id = f"{document_id}:{page['location']}:{index}"
                    stable_chunk_id = hashlib.sha256(
                        chunk_id.encode("utf-8")
                    ).hexdigest()

                    vector = embedding_model.encode(
                        text, normalize_embeddings=True
                    ).tolist()

                    chunks_to_upload.append({
                        "id": stable_chunk_id,
                        "content": text,
                        "source_file": item["name"],
                        "source_url": item["source_url"],
                        "location": page["location"],
                        "chunk_id": str(index),
                        "document_id": document_id,
                        "content_vector": vector,
                    })

            if not chunks_to_upload:
                failures.append({
                    "file": item["name"],
                    "error": "No extractable text found",
                })
                continue

            # Remove old chunks for this same SharePoint file so that
            # shorter updated versions do not leave stale extra chunks.
            old_chunks = search_client.search(
                search_text="*",
                filter=f"document_id eq '{document_id}'",
                select=["id"],
                top=1000,
            )
            old_ids = [{"id": row["id"]} for row in old_chunks]

            for start in range(0, len(old_ids), 100):
                search_client.delete_documents(
                    documents=old_ids[start:start + 100]
                )

            for start in range(0, len(chunks_to_upload), 100):
                batch = chunks_to_upload[start:start + 100]
                results = search_client.upload_documents(
                    documents=batch
                )

                for result in results:
                    if result.succeeded:
                        uploaded += 1
                    else:
                        failures.append({
                            "file": item["name"],
                            "error": result.error_message,
                        })

        except Exception as exc:
            failures.append({
                "file": item["name"],
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
    create_index()

    for file_path in files:
        try:
            document_id = stable_document_id(file_path)
            chunks_to_upload = []

            for page in extract_text(file_path):
                for text in chunk_text(page["text"]):
                    if not text.strip():
                        continue

                    vector = embedding_model.encode(
                        text, normalize_embeddings=True
                    ).tolist()

                    chunks_to_upload.append({
                        "id": hashlib.sha256(f"{document_id}:{page['location']}:{len(chunks_to_upload)}".encode()).hexdigest(),
                        "content": text,
                        "source_file": file_path.name,
                        "source_url": "",
                        "location": page["location"],
                        "chunk_id": str(len(chunks_to_upload)),
                        "document_id": document_id,
                        "content_vector": vector,
                    })

            # Keep requests comfortably small.
            for start in range(0, len(chunks_to_upload), 100):
                batch = chunks_to_upload[start:start + 100]
                results = search_client.upload_documents(
                    documents=batch
                )

                for result in results:
                    if result.succeeded:
                        uploaded += 1
                    else:
                        failures.append({
                            "file": file_path.name,
                            "key": result.key,
                            "error": result.error_message,
                        })

        except Exception as exc:
            failures.append({
                "file": file_path.name,
                "error": str(exc),
            })

    return {
        "files_found": len(files),
        "chunks_uploaded": uploaded,
        "failures": failures,
    }


def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the most relevant document chunks using vector search."""
    if not query.strip():
        raise ValueError("Query cannot be empty")

    vector = embedding_model.encode(
        query, normalize_embeddings=True
    ).tolist()

    vector_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=[
            "content",
            "source_file",
            "source_url",
            "location",
            "chunk_id",
        ],
        top=top_k,
    )

    return [
        {
            "content": result["content"],
            "source_file": result["source_file"],
            "source_url": result.get("source_url", ""),
            "location": result["location"],
            "chunk_id": result["chunk_id"],
            "score": result.get("@search.score"),
        }
        for result in results
    ]