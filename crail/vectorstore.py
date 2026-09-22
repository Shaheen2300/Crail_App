"""FAISS index management.

This is where Phantom Context Hallucination is either prevented or allowed
to happen: add_documents() appends to whatever index already exists on disk
unless the caller explicitly clears it first. That's intentional here -
CRAIL is designed to catch contamination after the fact, so the app needs a
way to *reproduce* an uncleared, contaminated index on demand for testing.
"""
import os
import shutil
import stat
import time

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL, FAISS_INDEX_DIR, OPENAI_API_KEY


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def load_vectorstore() -> FAISS | None:
    if not (FAISS_INDEX_DIR / "index.faiss").exists():
        return None
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def add_documents(docs: list[Document]) -> FAISS:
    """Add docs to the existing index, or create one if none exists yet."""
    vs = load_vectorstore()
    if vs is None:
        vs = FAISS.from_documents(docs, get_embeddings())
    else:
        vs.add_documents(docs)
    vs.save_local(str(FAISS_INDEX_DIR))
    return vs


def _clear_readonly_and_retry(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def clear_index() -> None:
    """Deletes the FAISS index directory. Retries on transient Windows file
    locks (this project lives inside OneDrive, which briefly locks files
    while syncing, causing rmtree to fail with 'Access is denied' even
    though nothing in the app itself is holding the file open)."""
    if not FAISS_INDEX_DIR.exists():
        return

    last_error = None
    for attempt in range(5):
        try:
            shutil.rmtree(FAISS_INDEX_DIR, onerror=_clear_readonly_and_retry)
            return
        except OSError as e:
            last_error = e
            time.sleep(0.3 * (attempt + 1))

    # The index files themselves are what index_exists() checks for; if
    # they're gone, a leftover empty directory (the container itself
    # briefly locked) is harmless and safe to leave for the next write.
    if not any(FAISS_INDEX_DIR.iterdir()):
        return

    raise RuntimeError(
        f"Could not clear the FAISS index at {FAISS_INDEX_DIR} after "
        f"several retries ({last_error}). This folder is inside OneDrive, "
        f"which can briefly lock files while syncing. Wait a moment and "
        f"try again."
    ) from last_error


def index_exists() -> bool:
    return (FAISS_INDEX_DIR / "index.faiss").exists()
