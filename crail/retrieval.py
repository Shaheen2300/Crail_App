"""Retrieval step: MMR search, top_k=4 (see config.TOP_K)."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import BROAD_TOP_K, TOP_K

# similarity_search's `filter` fetches this many raw nearest neighbors
# FIRST, then filters down to the metadata match - it's a separate
# parameter from `k` (default fetch_k=20 regardless of k). Passing a
# larger `k` alone does nothing: found the hard way when a 29-chunk
# document sharing an index with a 228-chunk document returned zero
# filtered results even at k=300, because none of the real top-20 raw
# nearest neighbors happened to belong to the smaller document. Every
# filtered call in this module passes fetch_k explicitly because of this.
FILTER_FETCH_K = 500


def retrieve(vectorstore: FAISS, query: str, k: int = TOP_K) -> list[Document]:
    fetch_k = max(k * 5, 20)
    return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def _opening_chunk(vectorstore: FAISS, query: str, current_doc_id: str) -> Document | None:
    candidates = vectorstore.similarity_search(
        query, k=1, fetch_k=FILTER_FETCH_K, filter={"source_doc": current_doc_id, "chunk_index": 0}
    )
    return candidates[0] if candidates else None


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

    opening = _opening_chunk(vectorstore, query, current_doc_id)
    if opening is None:
        return general_hits

    already_present = any(
        d.metadata.get("source_doc") == current_doc_id and d.metadata.get("chunk_index") == 0
        for d in general_hits
    )
    if already_present:
        return general_hits
    return [opening] + general_hits[: k - 1]


def retrieve_scoped(vectorstore: FAISS, query: str, current_doc_id: str, k: int = BROAD_TOP_K) -> list[Document]:
    """A fresh search restricted entirely to current_doc_id, ignoring
    whatever the original (possibly mixed-corpus) retrieval happened to
    surface. Used to build a Tier 2/3 review-queue suggestion from the
    best available verified-safe content, rather than being stuck with
    however few "matches current document" chunks survived the original
    top-k ranking against the whole index. If contamination was severe
    (e.g. 9 of 10 original chunks from the wrong document), the correct
    document's own best content - methods, results, conclusion - is very
    likely still sitting in the index; it just lost that ranking against
    a much larger or more query-relevant other document. This goes and
    gets it directly instead.
    """
    hits = vectorstore.similarity_search(
        query, k=k, fetch_k=FILTER_FETCH_K, filter={"source_doc": current_doc_id}
    )
    if not hits:
        return []

    opening = _opening_chunk(vectorstore, query, current_doc_id)
    if opening is None or any(d.metadata.get("chunk_index") == 0 for d in hits):
        return hits[:k]
    return [opening] + [d for d in hits if d.metadata.get("chunk_index") != 0][: k - 1]
