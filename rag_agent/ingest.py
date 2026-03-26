from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import Settings
from .embeddings import build_bedrock_embeddings
from .prompts import get_document_descriptor


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    document_name: str
    source_path: Path
    text: str


@dataclass(frozen=True)
class ChunkRecord:
    vector_id: str
    document_id: str
    document_name: str
    chunk_id: str
    text: str
    source: str
    section: str | None = None


def discover_documents(documents_dir: str | Path) -> list[DocumentRecord]:
    base_dir = Path(documents_dir)
    records: list[DocumentRecord] = []
    for source_path in sorted(base_dir.glob("*.pdf")):
        text = extract_pdf_text(source_path)
        if not text:
            continue
        descriptor = get_document_descriptor(source_path)
        records.append(
            DocumentRecord(
                document_id=descriptor["document_id"],
                document_name=descriptor["document_name"],
                source_path=source_path,
                text=text,
            )
        )
    return records


def extract_pdf_text(path: str | Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        cleaned = " ".join(page_text.split())
        if cleaned:
            parts.append(cleaned)
    return "\n".join(parts).strip()


def split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def chunk_document(
    document: DocumentRecord, *, chunk_size: int, chunk_overlap: int
) -> list[ChunkRecord]:
    chunks = split_text(
        document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return [
        ChunkRecord(
            vector_id=f"{document.document_id}:{index}",
            document_id=document.document_id,
            document_name=document.document_name,
            chunk_id=f"chunk_{index}",
            text=chunk,
            source=str(document.source_path),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def build_vectors(
    chunks: Iterable[ChunkRecord], embeddings_model
) -> list[tuple[str, list[float], dict[str, object]]]:
    chunk_list = list(chunks)
    if not chunk_list:
        return []

    vectors = embeddings_model.embed_documents([chunk.text for chunk in chunk_list])
    result = []
    for chunk, vector in zip(chunk_list, vectors, strict=True):
        metadata: dict[str, object] = {
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source": chunk.source,
        }
        if chunk.section is not None:
            metadata["section"] = chunk.section

        result.append(
            (
                chunk.vector_id,
                vector,
                metadata,
            )
        )
    return result


def ingest_documents(settings: Settings, documents_dir: str | Path) -> dict[str, int]:
    from pinecone import Pinecone

    documents = discover_documents(documents_dir)
    if not documents:
        raise ValueError(f"No PDF documents found in {documents_dir}")

    embeddings = build_bedrock_embeddings(settings)
    client = Pinecone(api_key=settings.pinecone_api_key)
    index = client.Index(settings.pinecone_index_name)

    all_chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    ]
    vectors = build_vectors(all_chunks, embeddings)
    if vectors:
        index.upsert(vectors=vectors, namespace=settings.pinecone_namespace)

    return {
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "vector_count": len(vectors),
    }
