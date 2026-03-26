from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is optional until install time
    def load_dotenv(_env_file: str | None = None) -> bool:
        return False


class SettingsError(ValueError):
    """Raised when required environment variables are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_chat_model: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    aws_region: str
    poly_gpt_embedding_model_id: str
    retriever_top_k: int
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "Settings":
        load_dotenv(env_file)

        missing = [
            name
            for name in (
                "OPENAI_API_KEY",
                "PINECONE_API_KEY",
                "PINECONE_INDEX_NAME",
                "POLY_GPT_EMBEDDING_MODEL_ID",
            )
            if not os.getenv(name)
        ]
        if missing:
            joined = ", ".join(missing)
            raise SettingsError(f"Missing required environment variables: {joined}")

        retriever_top_k = _get_int("RETRIEVER_TOP_K", default=4, minimum=1)
        chunk_size = _get_int("CHUNK_SIZE", default=800, minimum=100)
        chunk_overlap = _get_int("CHUNK_OVERLAP", default=120, minimum=0)
        if chunk_overlap >= chunk_size:
            raise SettingsError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

        return cls(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
            pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            aws_region=os.getenv("AWS_REGION", "ap-south-1"),
            poly_gpt_embedding_model_id=os.environ["POLY_GPT_EMBEDDING_MODEL_ID"],
            retriever_top_k=retriever_top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def _get_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise SettingsError(f"{name} must be at least {minimum}, got {value}")
    return value
