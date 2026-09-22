"""PDF -> text -> tagged chunks.

Every chunk is stamped with its source document at creation time
(metadata["source_doc"]). That stamp is a hard label set once, here, and is
never re-derived later from the chunk's content; that's what lets the
provenance check in crail.provenance be an objective measurement instead of
a judgment call.
"""
from io import BytesIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def chunk_document(text: str, doc_id: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    raw_chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={"source_doc": doc_id, "chunk_index": i},
        )
        for i, chunk in enumerate(raw_chunks)
    ]


def ingest_pdf(file_bytes: bytes, doc_id: str) -> list[Document]:
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        raise ValueError(
            f"No extractable text found in '{doc_id}'. It may be a scanned "
            "image PDF with no text layer."
        )
    return chunk_document(text, doc_id)
