"""Sentence-level attribution: which retrieved chunk (and which source
document) does each sentence of the answer most resemble?

This turns the single scalar contamination_ratio into something visible
inside the actual answer text - a reviewer (or a demo audience) can see
*which sentence* is riding on a chunk from the wrong document, not just
that some unspecified fraction of context was contaminated.

Deliberately a nearest-neighbor embedding match, not a claim of causal
attribution (the model may have blended multiple chunks into one
sentence) - it's a best-effort visualization aid, not a formal citation
system.
"""
import re

import numpy as np
from langchain_core.documents import Document


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def attribute_sentences(
    answer_text: str,
    retrieved_docs: list[Document],
    current_doc_id: str,
    embeddings,
    min_similarity: float = 0.35,
) -> list[dict]:
    """Returns a list of {sentence, source_doc, match, similarity} dicts,
    one per sentence. source_doc is None when no retrieved chunk is a
    close enough match (min_similarity) to attribute it."""
    sentences = split_sentences(answer_text)
    if not sentences or not retrieved_docs:
        return [{"sentence": s, "source_doc": None, "match": None, "similarity": 0.0} for s in sentences]

    chunk_vectors = np.array(embeddings.embed_documents([d.page_content for d in retrieved_docs]))
    sentence_vectors = np.array(embeddings.embed_documents(sentences))

    results = []
    for sentence, s_vec in zip(sentences, sentence_vectors):
        sims = [_cosine_sim(s_vec, c_vec) for c_vec in chunk_vectors]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        if best_sim < min_similarity:
            results.append({"sentence": sentence, "source_doc": None, "match": None, "similarity": round(best_sim, 3)})
        else:
            source_doc = retrieved_docs[best_idx].metadata.get("source_doc")
            results.append(
                {
                    "sentence": sentence,
                    "source_doc": source_doc,
                    "match": source_doc == current_doc_id,
                    "similarity": round(best_sim, 3),
                }
            )
    return results
