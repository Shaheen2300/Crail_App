"""Generation step: the LLM answers using only the retrieved chunks.

Deliberately does not hint the model to hedge or to phrase things a
particular way - CRAIL's confidence check (crail.confidence) needs to see
the model's natural phrasing, contaminated or not, for the risk signal to
mean anything.
"""
from langchain_core.documents import Document
from openai import OpenAI

from config import CHAT_MODEL, OPENAI_API_KEY

# One prompt per query intent (crail.intent.classify_intent). The model
# itself doesn't need to be smarter for summary/analysis requests, it
# already can do both, what it needed was permission and shape: told
# explicitly to synthesize across excerpts rather than locate one answer
# in them.
SYSTEM_PROMPTS = {
    "specific": (
        "You are a helpful assistant answering questions using only the "
        "excerpts provided below. Answer naturally and directly. If the "
        "excerpts don't contain the answer, say plainly that you can't "
        "find it in the document."
    ),
    "summary": (
        "You are a helpful assistant. Using only the excerpts provided "
        "below, write a clear, well-organized summary covering the "
        "document's purpose, key points, and main findings. Synthesize "
        "across all the excerpts rather than quoting just one. If the "
        "excerpts only cover part of the document, summarize what they "
        "cover and say the summary may be partial."
    ),
    "analysis": (
        "You are a helpful assistant. Using only the excerpts provided "
        "below, analyze the content: identify key findings, patterns, "
        "conclusions, and their implications. Synthesize across all the "
        "excerpts rather than quoting just one. If the excerpts don't "
        "contain enough to fully answer, say what's missing."
    ),
}
SYSTEM_PROMPT = SYSTEM_PROMPTS["specific"]  # kept for external callers/back-compat


def build_context_from_texts(texts: list[str]) -> str:
    return "\n\n".join(f"[Excerpt {i}]\n{t}" for i, t in enumerate(texts, start=1))


def build_context(retrieved_docs: list[Document]) -> str:
    return build_context_from_texts([doc.page_content for doc in retrieved_docs])


def generate_answer_from_texts(
    query: str, texts: list[str], client: OpenAI | None = None, intent: str = "specific"
) -> str:
    client = client or OpenAI(api_key=OPENAI_API_KEY)
    context = build_context_from_texts(texts)
    user_prompt = f"Excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
    result = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["specific"])},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return result.choices[0].message.content.strip()


def generate_answer(
    query: str, retrieved_docs: list[Document], client: OpenAI | None = None, intent: str = "specific"
) -> str:
    return generate_answer_from_texts(query, [d.page_content for d in retrieved_docs], client, intent=intent)
