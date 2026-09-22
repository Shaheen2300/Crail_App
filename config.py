"""Central configuration for CRAIL. All tunable constants live here so the
tier-reachability math (see README) can be checked in one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CRAIL_CHAT_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("CRAIL_JUDGE_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("CRAIL_EMBEDDING_MODEL", "text-embedding-3-small")

# --- RAG pipeline ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# Used for summary/analysis-intent queries (crail/intent.py), which need
# wide context rather than a narrow top-k similarity match. See
# crail/rri.py: the Tier 2 threshold is scaled down proportionally
# whenever more than TOP_K chunks are retrieved, so the "any single wrong
# chunk gets caught" guarantee holds at this k too, not just at TOP_K.
BROAD_TOP_K = 10

# --- RRI formula ---
CONTAMINATION_WEIGHT = 0.6
CONFIDENCE_WEIGHT = 0.4

# Was 0.25 - sat exactly on top of the smallest non-zero contamination value
# possible at TOP_K=4 (1/4 = 0.25), so a single stray chunk with a confident
# answer never qualified for the confidence bonus (0.25 is not > 0.25).
# 0.0 means any non-zero contamination counts; fully clean answers
# (contamination == 0) are still never affected since 0 is not > 0.
CONFIDENCE_MISMATCH_CONTAMINATION_FLOOR = 0.0

# --- Tier thresholds ---
# Was 0.3. At TOP_K=4, one bad chunk out of four gives
# RRI = 0.25 * CONTAMINATION_WEIGHT = 0.15 on contamination alone, with zero
# help from confidence detection. 0.3 sat above that, so a single-chunk leak
# was a guaranteed silent miss regardless of how the answer was phrased.
# 0.15 = the exact RRI of the smallest possible non-zero contamination at
# the current TOP_K, so ANY measurable contamination now reaches at least
# Tier 2 even if confidence detection contributes nothing at all. If you
# change TOP_K, recompute this: TIER2_THRESHOLD should stay at or below
# CONTAMINATION_WEIGHT * (1 / TOP_K).
TIER2_THRESHOLD = 0.15
TIER3_THRESHOLD = 0.7

# --- Storage paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
DOC_REGISTRY_PATH = DATA_DIR / "doc_registry.json"
REVIEW_QUEUE_PATH = DATA_DIR / "review_queue.json"
AUDIT_LOG_PATH = LOGS_DIR / "audit_log.jsonl"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
