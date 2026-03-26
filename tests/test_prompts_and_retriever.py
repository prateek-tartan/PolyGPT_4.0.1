from __future__ import annotations

import json
import unittest

from rag_agent.prompts import DOCUMENT_REGISTRY, build_system_prompt
from rag_agent.retriever import RetrievedChunk, format_chunks


class PromptAndRetrieverTests(unittest.TestCase):
    def test_system_prompt_contains_all_document_summaries(self) -> None:
        prompt = build_system_prompt()
        self.assertEqual(len(DOCUMENT_REGISTRY), 10)
        for document in DOCUMENT_REGISTRY:
            self.assertIn(document["document_name"], prompt)
            self.assertIn(document["document_id"], prompt)

    def test_format_chunks_returns_documents_used(self) -> None:
        payload = format_chunks(
            [
                RetrievedChunk(
                    document_id="doc_01",
                    document_name="Company Overview",
                    chunk_id="chunk_1",
                    text="Company Overview content",
                    score=0.91,
                ),
                RetrievedChunk(
                    document_id="doc_02",
                    document_name="Pricing Guide",
                    chunk_id="chunk_2",
                    text="Pricing details",
                    score=0.83,
                ),
            ]
        )
        parsed = json.loads(payload)
        self.assertEqual(parsed["documents_used"], ["Company Overview", "Pricing Guide"])
        self.assertEqual(len(parsed["matches"]), 2)

    def test_format_chunks_handles_empty_results(self) -> None:
        payload = format_chunks([])
        parsed = json.loads(payload)
        self.assertEqual(parsed["matches"], [])
        self.assertEqual(parsed["documents_used"], [])


if __name__ == "__main__":
    unittest.main()

