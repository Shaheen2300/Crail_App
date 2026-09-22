"""Tracks which documents have been added to the index, across sessions."""
import json
from datetime import datetime, timezone

from config import DOC_REGISTRY_PATH


def load_registry() -> list[dict]:
    if not DOC_REGISTRY_PATH.exists():
        return []
    return json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(entries: list[dict]) -> None:
    DOC_REGISTRY_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def register_document(doc_id: str, num_chunks: int, index_was_cleared: bool) -> None:
    entries = load_registry()
    if index_was_cleared:
        entries = []
    entries.append(
        {
            "doc_id": doc_id,
            "num_chunks": num_chunks,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_registry(entries)


def known_doc_ids() -> list[str]:
    seen = []
    for entry in load_registry():
        if entry["doc_id"] not in seen:
            seen.append(entry["doc_id"])
    return seen


def clear_registry() -> None:
    save_registry([])
