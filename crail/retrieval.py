"""Retrieval step: MMR search, top_k=4 (see config.TOP_K)."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import BROAD_TOP_K, TOP_K


def retrieve(vectorstore: FAISS, query: str, k: int = TOP_K) -> list[Document]:
    fetch_k = max(k * 5, 20)
    return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def retrieve_broad(vectorstore: FAISS, query: str, current_doc_id: str, k: int = BROAD_TOP_K) -> list[Document]:
    """For summary/analysis-intent queries. A plain top-k similarity match
    on a broad query ("summarize this") can be fooled by literal keyword
    overlap elsewhere in the corpus (a query containing the word "summary"
    can lose to a chunk literally titled "Summary Statistics" in an
    unrelated document) and starve out the document's own opening section,
    which is almost always where the title/abstract/executive-summary
    content lives. So this always includes that opening chunk, then fills
    the rest with the normal top-k MMR search.
    """
    general_hits = retrieve(vectorstore, query, k=k)

    opening_candidates = vectorstore.similarity_search(
        query, k=50, filter={"source_doc": current_doc_id, "chunk_index": 0}
    )
    if not opening_candidates:
        return general_hits

    opening = opening_candidates[0]
    already_present = any(
        d.metadata.get("source_doc") == current_doc_id and d.metadata.get("chunk_index") == 0
        for d in general_hits
    )
    if already_present:
        return general_hits
    return [opening] + general_hits[: k - 1]
