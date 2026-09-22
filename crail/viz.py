"""Embedding-space extraction for the visualization tab.

Pulls every stored vector straight out of the FAISS index (no re-embedding)
and reduces to 2D with PCA. PCA rather than t-SNE/UMAP deliberately: chunk
counts in a demo index are small (tens, not thousands), where t-SNE's
perplexity parameter is unstable and UMAP adds a heavy dependency for no
real benefit at this scale. PCA is deterministic and its two axes are
principal components of the actual embedding variance, which is enough to
show clustering by document.
"""
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.decomposition import PCA


def get_all_chunks_with_vectors(vs: FAISS) -> tuple[list[Document], np.ndarray]:
    n = vs.index.ntotal
    vectors = vs.index.reconstruct_n(0, n)
    docs = [vs.docstore.search(vs.index_to_docstore_id[i]) for i in range(n)]
    return docs, vectors


def project_2d(vectors: np.ndarray) -> np.ndarray:
    n_components = min(2, vectors.shape[0], vectors.shape[1])
    if n_components < 2:
        # Not enough points/dims for a 2D projection; pad with zeros.
        reduced = PCA(n_components=n_components).fit_transform(vectors)
        return np.pad(reduced, ((0, 0), (0, 2 - n_components)))
    return PCA(n_components=2).fit_transform(vectors)
