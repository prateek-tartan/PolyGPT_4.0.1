from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from rag_agent.config import Settings, SettingsError


class SettingsTests(unittest.TestCase):
    def test_missing_required_keys_raises_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SettingsError) as ctx:
                Settings.from_env(env_file=None)
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        self.assertIn("PINECONE_API_KEY", str(ctx.exception))

    def test_invalid_chunk_overlap_is_rejected(self) -> None:
        env = {
            "OPENAI_API_KEY": "openai",
            "PINECONE_API_KEY": "pinecone",
            "PINECONE_INDEX_NAME": "idx",
            "CHUNK_SIZE": "100",
            "CHUNK_OVERLAP": "100",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SettingsError) as ctx:
                Settings.from_env(env_file=None)
        self.assertIn("CHUNK_OVERLAP", str(ctx.exception))

    def test_env_defaults_are_applied(self) -> None:
        env = {
            "OPENAI_API_KEY": "openai",
            "PINECONE_API_KEY": "pinecone",
            "PINECONE_INDEX_NAME": "idx",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env(env_file=None)
        self.assertEqual(settings.openai_chat_model, "gpt-4o-mini")
        self.assertEqual(settings.retriever_top_k, 4)


if __name__ == "__main__":
    unittest.main()

