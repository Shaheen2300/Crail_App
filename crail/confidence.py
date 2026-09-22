"""Step 2: Confidence check.

Two interchangeable detectors for whether a response *sounds* like it's
confidently citing a source - the "illusion of epistemic grounding" that
makes contamination dangerous. Neither one changes whether a query gets
intercepted at all (that's driven entirely by contamination_ratio); this
only affects the tier2-vs-tier3 (hold-vs-block) decision. See README for why
that's a structural property of the RRI formula, not a bug in either
detector.

llm_judge is the default based on real evaluation data, not a guess: across
two separate runs (n=50 and n=150 contaminated queries), the keyword method
agreed with the llm_judge verdict on tier severity 0 times. gpt-4o-mini
answers factual questions plainly ("The total revenue was $57.6 million")
without ever using a listed trigger phrase, so keyword matching is close to
blind on modern model output. It's kept available for comparison, not
because it's competitive.
"""
from openai import OpenAI

from config import CONFIDENCE_MISMATCH_CONTAMINATION_FLOOR, JUDGE_MODEL, OPENAI_API_KEY

# Broadened from the original reference list. That original list missed real
# model phrasing in testing ("the document describes", "the document
# discusses", and mid-sentence "...provided in the document"). This list is
# still necessarily incomplete - any fixed phrase list will have gaps.
CONFIDENCE_PHRASES = [
    "according to the document",
    "the document states",
    "the text states",
    "based on the document",
    "as documented",
    "the document confirms",
    "the document shows",
    "as mentioned in",
    "the document indicates",
    "per the document",
    "the document reveals",
    "it states in the document",
    "the document describes",
    "the document discusses",
    "the document explains",
    "the document outlines",
    "the document mentions",
    "in the document",
    "found in the document",
    "documented in the text",
    "the text describes",
    "the text discusses",
    "the text explains",
]

JUDGE_PROMPT_TEMPLATE = """You are evaluating a single AI-generated answer for its
rhetorical confidence, not its correctness.

Answer to evaluate:
\"\"\"{response_text}\"\"\"

Question: Does this answer present its claims as confident, factual, and
grounded in a source document, regardless of whether the source is actually
correct? Or does it hedge, express uncertainty, or say it cannot find the
information?

Respond with exactly one word: CONFIDENT or HEDGED."""


def keyword_confidence(response_text: str) -> tuple[bool, int]:
    text = response_text.lower()
    count = sum(1 for p in CONFIDENCE_PHRASES if p in text)
    return count > 0, count


def llm_judge_confidence(response_text: str, client: OpenAI | None = None) -> bool:
    client = client or OpenAI(api_key=OPENAI_API_KEY)
    try:
        result = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT_TEMPLATE.format(response_text=response_text),
                }
            ],
            temperature=0,
            max_tokens=5,
        )
        verdict = result.choices[0].message.content.strip().upper()
        return verdict.startswith("CONFIDENT")
    except Exception:
        is_confident, _ = keyword_confidence(response_text)
        return is_confident


def check_confidence_mismatch(
    response_text: str,
    contamination_ratio: float,
    method: str = "llm_judge",
    client: OpenAI | None = None,
) -> tuple[bool, int]:
    """Returns (mismatch, phrase_count). phrase_count is 0/1 for the llm_judge
    method since it doesn't count phrases, only renders a verdict."""
    if method == "llm_judge":
        is_confident = llm_judge_confidence(response_text, client)
        phrase_count = 1 if is_confident else 0
    else:
        is_confident, phrase_count = keyword_confidence(response_text)

    mismatch = is_confident and contamination_ratio > CONFIDENCE_MISMATCH_CONTAMINATION_FLOOR
    return mismatch, phrase_count
