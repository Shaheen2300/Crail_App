"""Background audit log. Tier 1 responses are logged silently here; Tier 2/3
responses are logged here AND placed in the review queue."""
import json
from datetime import datetime, timezone

from config import AUDIT_LOG_PATH


def append_log(record: dict) -> None:
    record = {**record, "logged_at": datetime.now(timezone.utc).isoformat()}
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_log() -> list[dict]:
    if not AUDIT_LOG_PATH.exists():
        return []
    records = []
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clear_log() -> None:
    if AUDIT_LOG_PATH.exists():
        AUDIT_LOG_PATH.unlink()
