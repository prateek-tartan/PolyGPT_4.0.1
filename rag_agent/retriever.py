from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from .config import Settings
from .embeddings import build_bedrock_embeddings


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    document_name: str
    chunk_id: str
    text: str
    score: float | None = None
    section: str | None = None
    source: str | None = None


def format_chunks(chunks: Iterable[RetrievedChunk]) -> str:
    materialized = list(chunks)
    if not materialized:
        return json.dumps({"matches": [], "documents_used": []}, ensure_ascii=True)

    documents_used = sorted({chunk.document_name for chunk in materialized})
    matches = [
        {
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "chunk_id": chunk.chunk_id,
            "score": chunk.score,
            "section": chunk.section,
            "source": chunk.source,
            "text": chunk.text,
        }
        for chunk in materialized
    ]
    return json.dumps(
        {"matches": matches, "documents_used": documents_used},
        ensure_ascii=True,
        indent=2,
    )


class PineconeRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._embeddings = None
        self._index = None

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = build_bedrock_embeddings(self.settings)
        return self._embeddings

    def _get_index(self):
        if self._index is None:
            from pinecone import Pinecone

            client = Pinecone(api_key=self.settings.pinecone_api_key)
            self._index = client.Index(self.settings.pinecone_index_name)
        return self._index

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        embeddings = self._get_embeddings()
        index = self._get_index()
        vector = embeddings.embed_query(query)

        query_payload: dict[str, object] = {
            "vector": vector,
            "top_k": top_k or self.settings.retriever_top_k,
            "namespace": self.settings.pinecone_namespace,
            "include_metadata": True,
        }
        if document_id:
            query_payload["filter"] = {"document_id": {"$eq": document_id}}

        response = index.query(**query_payload)
        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches", [])

        results: list[RetrievedChunk] = []
        for match in matches or []:
            metadata = getattr(match, "metadata", None)
            if metadata is None and isinstance(match, dict):
                metadata = match.get("metadata", {})

            score = getattr(match, "score", None)
            if score is None and isinstance(match, dict):
                score = match.get("score")

            results.append(
                RetrievedChunk(
                    document_id=str(metadata.get("document_id", "")),
                    document_name=str(metadata.get("document_name", "Unknown Document")),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    text=str(metadata.get("text", "")),
                    score=float(score) if score is not None else None,
                    section=_optional_str(metadata.get("section")),
                    source=_optional_str(metadata.get("source")),
                )
            )
        return results


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
