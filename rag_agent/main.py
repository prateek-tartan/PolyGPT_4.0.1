from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import build_agent, run_agent
from .config import Settings, SettingsError
from .ingest import ingest_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph RAG agent with Pinecone")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Run the agent for a user query")
    ask_parser.add_argument("query", help="The user question to answer")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Chunk local documents and upsert them to Pinecone"
    )
    ingest_parser.add_argument(
        "--documents-dir",
        default=str(Path("data") / "documents"),
        help="Directory containing the curated documents",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = Settings.from_env()
    except SettingsError as exc:
        print(f"Configuration error: {exc}")
        return 1

    if args.command == "ingest":
        result = ingest_documents(settings, args.documents_dir)
        print(json.dumps(result, indent=2))
        return 0

    agent = build_agent(settings)
    result = run_agent(agent, args.query)
    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
