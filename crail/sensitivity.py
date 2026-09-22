"""Threshold and weight sensitivity analysis, computed from a saved
evaluate.py run (eval_results.json). Pure arithmetic on already-collected
contamination_ratio / confidence_mismatch values - no new API calls -
because RRI is a deterministic function of those two inputs, so we can
replay the formula under different hyperparameters without regenerating
anything.
"""
from config import CONFIDENCE_WEIGHT, CONTAMINATION_WEIGHT, TIER2_THRESHOLD

# evaluate.py has used two naming schemes across revisions of the harness:
# "contaminated"/"clean" (paired-document runs) and "shared"/"isolated"
# (shared-index growth runs). Both are accepted so old and new
# eval_results.json files feed the same sweep.
CONTAMINATED_LABELS = {"contaminated", "shared"}
CLEAN_LABELS = {"clean", "isolated"}


def _rri(contamination_ratio: float, confidence_mismatch: bool, contamination_weight: float, confidence_weight: float) -> float:
    return min(contamination_ratio * contamination_weight + (confidence_weight if confidence_mismatch else 0.0), 1.0)


def sweep_tier2_threshold(
    results: list[dict],
    thresholds: list[float],
    contamination_weight: float = CONTAMINATION_WEIGHT,
    confidence_weight: float = CONFIDENCE_WEIGHT,
) -> list[dict]:
    contaminated = [r for r in results if r["scenario"] in CONTAMINATED_LABELS]
    clean = [r for r in results if r["scenario"] in CLEAN_LABELS]
    rows = []
    for t in thresholds:
        intercepted = sum(
            1
            for r in contaminated
            if _rri(r["contamination_ratio"], r["confidence_mismatch_keyword"], contamination_weight, confidence_weight) >= t
        )
        false_positives = sum(
            1
            for r in clean
            if _rri(r["contamination_ratio"], r["confidence_mismatch_keyword"], contamination_weight, confidence_weight) >= t
        )
        rows.append(
            {
                "threshold": round(t, 3),
                "interception_rate": intercepted / len(contaminated) if contaminated else 0.0,
                "false_positive_rate": false_positives / len(clean) if clean else 0.0,
            }
        )
    return rows


def sweep_contamination_weight(
    results: list[dict],
    weights: list[float],
    tier2_threshold: float = TIER2_THRESHOLD,
    confidence_weight: float = CONFIDENCE_WEIGHT,
) -> list[dict]:
    contaminated = [r for r in results if r["scenario"] in CONTAMINATED_LABELS]
    clean = [r for r in results if r["scenario"] in CLEAN_LABELS]
    rows = []
    for w in weights:
        intercepted = sum(
            1
            for r in contaminated
            if _rri(r["contamination_ratio"], r["confidence_mismatch_keyword"], w, confidence_weight) >= tier2_threshold
        )
        false_positives = sum(
            1
            for r in clean
            if _rri(r["contamination_ratio"], r["confidence_mismatch_keyword"], w, confidence_weight) >= tier2_threshold
        )
        rows.append(
            {
                "contamination_weight": round(w, 3),
                "interception_rate": intercepted / len(contaminated) if contaminated else 0.0,
                "false_positive_rate": false_positives / len(clean) if clean else 0.0,
            }
        )
    return rows
