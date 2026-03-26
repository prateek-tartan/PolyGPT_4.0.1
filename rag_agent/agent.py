from __future__ import annotations

import json
import sys
from typing import Any

from .config import Settings
from .prompts import build_system_prompt
from .retriever import PineconeRetriever, format_chunks


def build_retriever_tool(retriever: PineconeRetriever):
    from langchain_core.tools import tool

    @tool
    def retrieve_documents(query: str, top_k: int, document_id: str | None = None) -> str:
        """Search Pinecone for relevant document chunks.

        Args:
            query: The focused retrieval query to search for.
            top_k: Required number of chunks to retrieve. Choose it deliberately for
                each call instead of reusing a default. Use 2 for pinpoint factual
                lookups, 3-4 for normal document-specific questions, 5-6 for
                multi-part or synthesis questions within one document, and 7-8 for
                broad comparison or cross-document synthesis questions.
            document_id: Optional document identifier to restrict retrieval to one
                specific document when the user's question clearly targets it.
        """

        log_payload = {"query": query, "top_k": top_k, "document_id": document_id}
        print(
            f"[tool-call] retrieve_documents {json.dumps(log_payload, ensure_ascii=True)}",
            file=sys.stderr,
        )
        chunks = retriever.search(query, top_k=top_k, document_id=document_id)
        formatted = format_chunks(chunks)
        try:
            parsed = json.loads(formatted)
            documents_used = parsed.get("documents_used", [])
            match_count = len(parsed.get("matches", []))
            print(
                "[tool-result] retrieve_documents "
                f"{json.dumps({'match_count': match_count, 'documents_used': documents_used}, ensure_ascii=True)}",
                file=sys.stderr,
            )
        except json.JSONDecodeError:
            print("[tool-result] retrieve_documents", file=sys.stderr)
        return formatted

    return retrieve_documents


def build_agent(settings: Settings):
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    retriever = PineconeRetriever(settings)
    model = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
    )
    return create_react_agent(
        model=model,
        tools=[build_retriever_tool(retriever)],
        prompt=build_system_prompt(),
    )


def run_agent(agent: Any, user_query: str) -> dict[str, Any]:
    response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
    messages = response.get("messages", [])
    final_message = messages[-1].content if messages else ""
    return {"answer": final_message, "messages": messages}
