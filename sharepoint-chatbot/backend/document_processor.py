from pathlib import Path
import hashlib
import fitz
from pptx import Presentation


SUPPORTED_EXTENSIONS = {".pdf", ".pptx"}


def extract_text(file_path: Path) -> list[dict]:
    """Extract text while preserving page/slide references."""
    pages = []

    if file_path.suffix.lower() == ".pdf":
        with fitz.open(file_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({
                        "location": f"Page {page_number}",
                        "text": text,
                    })

    elif file_path.suffix.lower() == ".pptx":
        presentation = Presentation(file_path)

        for slide_number, slide in enumerate(
            presentation.slides, start=1
        ):
            parts = []

            for shape in slide.shapes:
                if shape.has_text_frame and shape.text.strip():
                    parts.append(shape.text.strip())

                if shape.has_table:
                    for row in shape.table.rows:
                        cells = [
                            cell.text.strip() for cell in row.cells
                        ]
                        parts.append(" | ".join(cells))

            text = "\n".join(parts).strip()
            if text:
                pages.append({
                    "location": f"Slide {slide_number}",
                    "text": text,
                })

    return pages


def chunk_text(text: str, chunk_size: int = 1000,
               overlap: int = 150) -> list[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap")

    words = text.split()
    chunks = []
    current_words = []
    current_length = 0

    for word in words:
        added_length = len(word) + (1 if current_words else 0)

        if current_words and current_length + added_length > chunk_size:
            chunks.append(" ".join(current_words))

            # Retain trailing words for overlap.
            tail = []
            tail_length = 0

            for previous_word in reversed(current_words):
                length = len(previous_word) + (
                    1 if tail else 0
                )
                if tail and tail_length + length > overlap:
                    break
                tail.insert(0, previous_word)
                tail_length += length

            current_words = tail
            current_length = len(" ".join(current_words))

        current_words.append(word)
        current_length += len(word) + (
            1 if len(current_words) > 1 else 0
        )

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def stable_document_id(file_path: Path) -> str:
    """Create a stable identifier for a file."""
    return hashlib.sha256(file_path.name.encode("utf-8")).hexdigest()