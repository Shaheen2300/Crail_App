"""Persisted queue for Tier 2 (hold) and Tier 3 (block) items awaiting a
human reviewer. This is the actual point of CRAIL, not set-dressing: a
detector with no usable review step is just a filter that silently drops
things.
"""
import json
import uuid
from datetime import datetime, timezone

from config import REVIEW_QUEUE_PATH


def load_queue() -> list[dict]:
    if not REVIEW_QUEUE_PATH.exists():
        return []
    return json.loads(REVIEW_QUEUE_PATH.read_text(encoding="utf-8"))


def save_queue(items: list[dict]) -> None:
    REVIEW_QUEUE_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def add_item(item: dict) -> str:
    items = load_queue()
    item_id = str(uuid.uuid4())
    item = {
        "id": item_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        **item,
    }
    items.append(item)
    save_queue(items)
    return item_id


def update_item(item_id: str, **fields) -> None:
    """Resolve an item: sets the given fields and stamps resolved_at."""
    items = load_queue()
    for item in items:
        if item["id"] == item_id:
            item.update(fields)
            item["resolved_at"] = datetime.now(timezone.utc).isoformat()
            break
    save_queue(items)


def patch_item(item_id: str, **fields) -> None:
    """Update fields on a still-pending item (e.g. a draft suggestion)
    without resolving it."""
    items = load_queue()
    for item in items:
        if item["id"] == item_id:
            item.update(fields)
            break
    save_queue(items)


def pending_items(tier: int | None = None) -> list[dict]:
    items = [i for i in load_queue() if i["status"] == "pending"]
    if tier is not None:
        items = [i for i in items if i["tier"] == tier]
    return items


def resolved_items() -> list[dict]:
    return [i for i in load_queue() if i["status"] != "pending"]


def clear_queue() -> None:
    save_queue([])
