# POC: Retriever As A Tool

## Goal

Build a RAG agent where retrieval is an explicit tool call instead of an automatic hidden step.

The LLM should decide:

- whether retrieval is needed
- which document to search
- what query to send
- how many chunks to retrieve using `top_k`

## Implementation

### Ingestion

PDFs from [`data/documents`](/Users/prateek/Polygpt%204.0.1/data/documents) are chunked, embedded, and stored in Pinecone with metadata such as:

- `document_id`
- `document_name`
- `chunk_id`
- `text`

### Prompt Registry

[`rag_agent/prompts.py`](/Users/prateek/Polygpt%204.0.1/rag_agent/prompts.py) builds a system prompt containing, for each document:

- document name
- document id
- short summary
- hint about when that document should be retrieved

This acts like a lightweight registry that helps the LLM map a user query to the right document.

### Tool

[`rag_agent/agent.py`](/Users/prateek/Polygpt%204.0.1/rag_agent/agent.py) exposes retrieval as:

```python
retrieve_documents(query: str, top_k: int, document_id: str | None = None) -> str
```

The tool:

- embeds the query
- searches Pinecone
- optionally filters by `document_id`
- returns matched chunks as JSON

### Runtime Logging

Tool activity is logged in the terminal:

```text
[tool-call] retrieve_documents {...}
[tool-result] retrieve_documents {...}
```

This makes retrieval decisions visible during agent execution.

## Key Behavior

The current setup supports:

- explicit tool-based retrieval
- LLM-selected `document_id` for targeted search
- LLM-selected `top_k` based on question scope
- source-grounded final answers

## Results

### 1. For difficult multi-document questions, the LLM decomposes the problem and makes multiple targeted retrieval calls

Observed behavior:

- the user asked one synthesis question spanning MSME, underwriting, and NMDFC policy
- the LLM issued multiple retriever calls
- each call used a different `document_id`
- each call used a document-specific query

Observed logs:

```text
[tool-call] retrieve_documents {"query": "MSME Short Term Finance Program Note market gap and collateral/information problems", "top_k": 3, "document_id": "msme_stl_page_2_3"}
[tool-call] retrieve_documents {"query": "ABCL Credit Underwriting Process Note for STSL and MLAP operational risk checks", "top_k": 3, "document_id": "process_note_on_credit_underwriting_for_stsl_and_mlap_v1"}
[tool-call] retrieve_documents {"query": "NMDFC Lending Policy eligibility control", "top_k": 3, "document_id": "national_minorities_development_and_finance_corporation"}
```

This shows that the model used the prompt registry to identify the right files and query them separately.

### 2. For ambiguous questions, the LLM increases `top_k` and performs broader retrieval across multiple files

Observed behavior:

- the question did not clearly point to one document
- the LLM did not pass `document_id`
- it used a higher `top_k`
- one tool call retrieved evidence from multiple documents

Observed logs:

```text
[tool-call] retrieve_documents {"query": "eligibility rules, risk checks, and approval controls for lending program", "top_k": 6, "document_id": null}
[tool-result] retrieve_documents {"match_count": 6, "documents_used": ["ABCL Credit Underwriting Process Note for STSL and MLAP", "NMDFC Lending Policy", "Two Wheeler Loans Product Approval Document"]}
```

This shows broader recall behavior when the query is not specific enough to map to one file.

### 3. For low-clarity questions, the LLM can ask for clarification instead of retrieving blindly

Observed behavior:

- the query was too vague to map to a specific document or topic
- the model asked a clarification question
- no unnecessary retrieval was performed first

Observed example:

```text
User: What are the important rules?
Assistant: Could you please specify which context or area you are referring to when you mention "important rules"? For example, are you asking about lending policies, professional loan rules, or something else?
```

This shows the system can avoid low-confidence retrieval when the request lacks enough context.

## Conclusion

This POC shows that retrieval can be treated as a deliberate tool call rather than a fixed preprocessing step.

The LLM is able to:

- choose when to retrieve
- choose whether to target one document or search broadly
- change `top_k` based on query complexity
- ask for clarification when the request is too vague
