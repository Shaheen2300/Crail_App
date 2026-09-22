"""Query intent classification.

The underlying model (gpt-4o-mini) is already fully capable of summarizing,
analyzing, and drawing conclusions - that was never the limitation. What
was actually starving those query types was retrieval: a narrow top-4
similarity search is well suited to a specific factual lookup ("what is
the rate limit"), but a broad request ("summarize this", "what are the
conclusions") needs wide context, and generic query words like "summary"
or "report" can literally out-rank the real content on embedding
similarity (see README's retrieval-quality note). This module decides
which retrieval/prompting mode a query needs; crail.retrieval and
crail.generation act on that decision.

Fast heuristic, no extra API call: coverage is good enough on common
phrasings that spending a model call to disambiguate isn't worth it, and
misses just fall back to "specific" (today's existing behavior), so this
is purely additive, never a regression.
"""
import re

SUMMARY_PATTERNS = re.compile(
    r"\b(summar(y|ize|ise|ies)|overview|tl;?dr|what.{0,15}(this|the)\s+(document|report|paper|file)\s+(is\s+)?about)\b",
    re.IGNORECASE,
)
ANALYSIS_PATTERNS = re.compile(
    r"\b(analy(z|s)e|analysis|conclusions?|key\s+(findings?|points?|takeaways?)|main\s+findings?|"
    r"insights?|implications?|what.{0,10}(does|did).{0,15}(find|conclude|show))\b",
    re.IGNORECASE,
)


def classify_intent(query: str) -> str:
    """Returns 'summary', 'analysis', or 'specific'."""
    if SUMMARY_PATTERNS.search(query):
        return "summary"
    if ANALYSIS_PATTERNS.search(query):
        return "analysis"
    return "specific"
