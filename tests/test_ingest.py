from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_agent.ingest import ChunkRecord, build_vectors, chunk_document, split_text


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(index), float(len(text))] for index, text in enumerate(texts, start=1)]


class IngestTests(unittest.TestCase):
    def test_split_text_returns_overlapping_chunks(self) -> None:
        text = "A" * 120 + " " + "B" * 120 + " " + "C" * 120
        chunks = split_text(text, chunk_size=150, chunk_overlap=25)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(chunk for chunk in chunks))

    def test_chunk_document_builds_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "doc.txt"
            path.write_text("Example text " * 100, encoding="utf-8")
            from rag_agent.ingest import DocumentRecord

            document = DocumentRecord(
                document_id="doc_01",
                document_name="Example Doc",
                source_path=path,
                text=path.read_text(encoding="utf-8"),
            )
            chunks = chunk_document(document, chunk_size=120, chunk_overlap=20)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].document_name, "Example Doc")
        self.assertEqual(chunks[0].chunk_id, "chunk_1")

    def test_build_vectors_preserves_chunk_metadata(self) -> None:
        chunks = [
            ChunkRecord(
                vector_id="doc_01:1",
                document_id="doc_01",
                document_name="Doc One",
                chunk_id="chunk_1",
                text="hello world",
                source="/tmp/doc1.txt",
            )
        ]
        vectors = build_vectors(chunks, FakeEmbeddings())
        vector_id, values, metadata = vectors[0]
        self.assertEqual(vector_id, "doc_01:1")
        self.assertEqual(values, [1.0, 11.0])
        self.assertEqual(metadata["document_name"], "Doc One")


if __name__ == "__main__":
    unittest.main()

