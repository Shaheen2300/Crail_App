"""Step 3 & 4: the Retrieval Risk Index formula and tier routing.

RRI = (contamination_ratio * CONTAMINATION_WEIGHT)
    + (CONFIDENCE_WEIGHT if confidence_mismatch else 0), capped at 1.0

Structural property to keep in mind: with CONTAMINATION_WEIGHT=0.6, a
response can never reach Tier 3 on contamination alone (max is 1.0*0.6=0.6,
below TIER3_THRESHOLD=0.7). Reaching Tier 3 always requires both high
contamination AND a detected confidence mismatch. If you change the weights
or thresholds in config.py, recheck this ceiling.

Second property, previously a real blind spot and now fixed: at TOP_K=4 the
smallest non-zero contamination is exactly 0.25 (one bad chunk out of four).
TIER2_THRESHOLD is set at or below CONTAMINATION_WEIGHT * 0.25 so that value
alone always reaches Tier 2, even with zero help from confidence detection.
Verified against real evaluation data: with the old 0.3 threshold, a
document pair with low semantic overlap produced exactly this case on every
query and was silently delivered as Tier 1 every single time. If you change
TOP_K, recompute this floor.

Third property: crail.retrieval.retrieve_broad returns more than TOP_K
chunks for summary/analysis-intent queries. effective_tier2_threshold()
scales the floor down proportionally (never up) so a single wrong chunk out
of N is always caught at any N, not just at TOP_K - the same guarantee,
generalized instead of silently only applying at the default k.
"""
from openai import OpenAI

from config import (
    CONFIDENCE_WEIGHT,
    CONTAMINATION_WEIGHT,
    TIER2_THRESHOLD,
    TIER3_THRESHOLD,
    TOP_K,
)
from crail.confidence import check_confidence_mismatch
from crail.provenance import check_provenance


def effective_tier2_threshold(num_retrieved: int) -> float:
    if num_retrieved <= TOP_K or num_retrieved <= 0:
        return TIER2_THRESHOLD
    return min(TIER2_THRESHOLD, CONTAMINATION_WEIGHT / num_retrieved)


def crail_decision(
    retrieved_docs,
    response_text: str,
    current_doc_id: str,
    confidence_method: str = "llm_judge",
    client: OpenAI | None = None,
) -> dict:
    contamination_ratio = check_provenance(retrieved_docs, current_doc_id)
    mismatch, phrase_count = check_confidence_mismatch(
        response_text, contamination_ratio, method=confidence_method, client=client
    )
    rri = min(
        (contamination_ratio * CONTAMINATION_WEIGHT)
        + (CONFIDENCE_WEIGHT if mismatch else 0.0),
        1.0,
    )
    tier2_threshold = effective_tier2_threshold(len(retrieved_docs))

    if rri >= TIER3_THRESHOLD:
        tier, flag, action = 3, "HIGH_RETRIEVAL_RISK", "BLOCK"
    elif rri >= tier2_threshold:
        tier, flag, action = 2, "MEDIUM_RETRIEVAL_RISK", "HOLD_FOR_REVIEW"
    else:
        tier, flag, action = 1, "CLEAN", "DELIVER"

    return {
        "tier": tier,
        "flag": flag,
        "rri": round(rri, 3),
        "contamination_ratio": round(contamination_ratio, 3),
        "confidence_mismatch": mismatch,
        "confidence_phrases": phrase_count,
        "confidence_method": confidence_method,
        "tier2_threshold": round(tier2_threshold, 4),
        "action": action,
    }
