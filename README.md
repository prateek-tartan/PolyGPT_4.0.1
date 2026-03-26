# LangGraph RAG Agent

Python LangGraph agent with a Pinecone-backed retriever tool. Retrieval is explicit: the agent reads a static 10-document summary prompt, decides when retrieval is necessary, and calls the retriever tool only when document grounding is needed.

## Setup

1. Create and populate `.env` from [.env.example](/Users/prateek/Downloads/Polygpt 4.0.1/.env.example).
2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Put the curated PDF documents in [data/documents](/Users/prateek/Downloads/Polygpt 4.0.1/data/documents) using filenames that start with the registered document IDs such as `doc_01_company_overview.pdf`.

## Run ingestion

```bash
python3 -m rag_agent.main ingest --documents-dir data/documents
```

This extracts text from the PDFs, chunks the content, embeds it with OpenAI embeddings, creates the Pinecone index if needed, and upserts the vectors with document metadata.

## Run the agent

```bash
python3 -m rag_agent.main ask "What does the pricing guide say about upgrade limits?"
```

The agent uses a system prompt that includes two lines per document and may call the Pinecone retriever tool. Final answers should include a `Sources:` line naming the documents actually used.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
