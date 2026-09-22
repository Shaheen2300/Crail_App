"""Security layer on the human review step itself.

CRAIL's tiering catches contaminated model output before it reaches the
user - but Tier 3 hands control to a human who can type anything into the
response box. This module closes that gap: before a human-composed
response is accepted as final, it's checked against verified (green,
current-document) content. A reviewer typing something unrelated or
fabricated gets rejected and must either revise or explicitly override
with a logged justification - never a silent pass-through.
"""
from openai import OpenAI

from config import JUDGE_MODEL, OPENAI_API_KEY

GROUNDING_PROMPT_TEMPLATE = """You are checking whether a human-written reply to a
user's question is acceptable to send, given verified excerpts from the
document the user is actually asking about.

Verified document excerpts:
\"\"\"{excerpts}\"\"\"

Human-written reply:
\"\"\"{response_text}\"\"\"

Question: is the reply one of the following?
(a) Consistent with and supported by the excerpts, OR
(b) A reasonable non-factual reply that makes no specific claim about the
document's content (for example: "I don't have that information",
"please contact support", or a request for clarification).

Reject only if the reply asserts a specific factual claim about the
document that the excerpts do NOT support or that they contradict, or if
the reply is about a clearly unrelated topic.

Respond with exactly one word: GROUNDED or REJECT."""


def check_grounding(
    response_text: str, excerpts: list[str], client: OpenAI | None = None
) -> tuple[bool, str]:
    """Returns (grounded, reason). Fails closed: any ambiguity or missing
    verified context rejects rather than silently accepting."""
    if not response_text.strip():
        return False, "Response is empty."

    if not excerpts:
        return (
            False,
            "No verified content exists for this document to check the response "
            "against. Use the override below if you've confirmed this manually.",
        )

    client = client or OpenAI(api_key=OPENAI_API_KEY)
    excerpt_block = "\n\n".join(f"[Excerpt {i + 1}]\n{e}" for i, e in enumerate(excerpts))
    try:
        result = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": GROUNDING_PROMPT_TEMPLATE.format(
                        excerpts=excerpt_block, response_text=response_text
                    ),
                }
            ],
            temperature=0,
            max_tokens=5,
        )
        verdict = result.choices[0].message.content.strip().upper()
        grounded = verdict.startswith("GROUNDED")
        reason = (
            "Consistent with verified document content."
            if grounded
            else "Not supported by this document's verified content."
        )
        return grounded, reason
    except Exception as e:
        return False, f"Grounding check couldn't run ({e}); failing closed."
