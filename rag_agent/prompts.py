from __future__ import annotations

from pathlib import Path
from textwrap import dedent


DOCUMENT_SUMMARY_BY_FILE = {
    "2 Wheeler Loan": {
        "document_name": "Two Wheeler Loans Product Approval Document",
        "summary": (
            "Internal product approval note covering the two-wheeler loan program, portfolio metrics, governance classification, and review cadence.\n"
            "Retrieve when the user asks about two-wheeler loan product design, performance benchmarks, approval details, or portfolio expectations."
        ),
    },
    "MSME_STL_page_1": {
        "document_name": "MSME Short Term Finance Program Cover Note",
        "summary": (
            "Opening page of the MSME short-term finance program note dated July 2018, identifying the program and source material.\n"
            "Retrieve when the user needs the program title, document identity, or the start of the MSME short-term finance note."
        ),
    },
    "MSME_STL_page_2_3": {
        "document_name": "MSME Short Term Finance Program Note",
        "summary": (
            "Describes the MSME financing opportunity in India, the market gap, borrower context, and why short-term finance is relevant.\n"
            "Retrieve when the user asks about the MSME segment, debt gap, borrower profile, or business rationale for the short-term finance program."
        ),
    },
    "National Minorities Development and Finance Corporation": {
        "document_name": "NMDFC Lending Policy",
        "summary": (
            "Formal lending policy of the National Minorities Development and Finance Corporation with institutional scope and updated policy guidance.\n"
            "Retrieve when the user asks about NMDFC policy rules, lending framework, institutional scope, or finance-corporation guidance."
        ),
    },
    "Process Note on  Credit underwriting for STSL and MLAP V1": {
        "document_name": "ABCL Credit Underwriting Process Note for STSL and MLAP",
        "summary": (
            "Process note for credit underwriting of STSL and MLAP covering validation, dedupe, bureau checks, inspections, collateral, queries, and eligibility assessment.\n"
            "Retrieve when the user asks about secured underwriting workflow, underwriting stages, operational checks, or credit-process responsibilities."
        ),
    },
    "Professional_Loan_Degree_Surrogate_Policy__v5": {
        "document_name": "ABFL Professional Loans Policy v5",
        "summary": (
            "Versioned product program policy for professional loans, including policy versioning, validity dates, and governing program rules.\n"
            "Retrieve when the user asks about professional loan policy, version changes, policy validity, or program-level loan rules."
        ),
    },
}


def infer_document_id(path: str | Path) -> str:
    stem = Path(path).stem.strip().lower()
    normalized = []
    last_was_underscore = False
    for char in stem:
        if char.isalnum():
            normalized.append(char)
            last_was_underscore = False
        else:
            if not last_was_underscore:
                normalized.append("_")
                last_was_underscore = True
    return "".join(normalized).strip("_")


def get_document_descriptor(path: str | Path) -> dict[str, str]:
    path_obj = Path(path)
    entry = DOCUMENT_SUMMARY_BY_FILE.get(path_obj.stem)
    if entry:
        return {
            "document_id": infer_document_id(path_obj),
            "document_name": entry["document_name"],
            "summary": entry["summary"],
        }

    inferred_name = path_obj.stem.replace("_", " ").strip()
    return {
        "document_id": infer_document_id(path_obj),
        "document_name": inferred_name,
        "summary": (
            "Uploaded source document available in the Pinecone knowledge base.\n"
            "Retrieve when the user query is likely tied to this document title or its domain."
        ),
    }


def build_document_registry(documents_dir: str | Path = "data/documents") -> list[dict[str, str]]:
    base_dir = Path(documents_dir)
    return [get_document_descriptor(path) for path in sorted(base_dir.glob("*.pdf"))]


def build_system_prompt(documents_dir: str | Path = "data/documents") -> str:
    document_registry = build_document_registry(documents_dir)
    document_lines = []
    for document in document_registry:
        document_lines.append(
            f"- {document['document_name']} ({document['document_id']}):\n{document['summary']}"
        )

    registry_block = "\n".join(document_lines) if document_lines else "- No PDF documents have been registered yet."
    return dedent(
        f"""
        You are a retrieval-aware assistant backed by a Pinecone document index.

        You have access to a retriever tool. Use it only when the user asks about facts
        that may live in the indexed documents. Do not call the retriever for casual chat,
        simple reasoning, or when the answer is already fully available without document support.

        Current document summaries:
        {registry_block}

        Retrieval rules:
        - Prefer the summaries above to decide whether retrieval is needed.
        - If a question is factual and document-backed, call the retriever tool.
        - You must choose `top_k` explicitly for every retrieval call. Do not treat `3`
          as a default.
        - Use `top_k=2` for pinpoint factual lookups or a single clause you expect to
          find in one chunk.
        - Use `top_k=3` or `top_k=4` for normal document-specific questions.
        - Use `top_k=5` or `top_k=6` for multi-part questions within one document or
          when you expect the evidence to span multiple chunks.
        - Use `top_k=7` or `top_k=8` for broad comparison, synthesis, or cross-document
          questions where you need wider recall.
        - Pass `document_id` when the question clearly refers to one specific document.
        - For cross-document questions, either make multiple targeted retrieval calls
          with document-specific `document_id` values, or use a larger `top_k` when a
          broader search is needed.
        - If the retriever returns no relevant chunks, say that the documents do not contain
          enough information instead of guessing.
        - In the final answer, include a `Sources:` line with the document names actually used.
        """
    ).strip()
