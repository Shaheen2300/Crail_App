"""Step 1: Provenance check (contamination ratio).

Measures what fraction of retrieved chunks did NOT come from the document
the user is currently querying, using the hard source_doc label stamped on
each chunk at ingestion time (see crail.ingestion). This is deliberately not
a content-similarity or relevance judgment - it's a metadata lookup, which
is what makes it objective and verifiable rather than a guess.
"""
from langchain_core.documents import Document


def check_provenance(retrieved_docs: list[Document], current_doc_id: str) -> float:
    if not retrieved_docs:
        return 0.0
    wrong = sum(
        1
        for doc in retrieved_docs
        if doc.metadata.get("source_doc", "UNKNOWN") != current_doc_id
    )
    return wrong / len(retrieved_docs)
