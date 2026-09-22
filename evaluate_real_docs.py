"""Unrelated-topic-query test on real, properly-sized documents.

Direct follow-up to evaluate_borderline.py's surprise finding: with tiny
2-3-chunk synthetic documents, even completely unrelated queries got held
for review (near-universal Tier 2 flagging), because the shared index was
barely larger than TOP_K=4 so MMR had no room to discriminate relevant
from irrelevant chunks. This script re-runs the same kind of test on real,
large, properly-sized documents (tens to hundreds of thousands of
characters each, hundreds of chunks) sourced from an existing CRAIL
research study (genai-chatbot/crail/data), to check whether that was a
real deployment risk or a small-test-corpus artifact.

Current document: pair1/doc_b, NIST AI 600-1 (Generative AI Risk
Management Framework Profile). Shared index also contains three fully
unrelated real document pairs: neonatal health / AI-in-nursing-care
(pair2_neonatal), RAG research papers (pair5), and GPT-3/GPT-4 papers
(pair6), none of which have anything to do with the NIST risk taxonomy
questions asked below.

Uses the real pipeline throughout (real PDF extraction, real embeddings,
real generation, real RRI scoring, current fixed thresholds from
config.py) and deliberately bypasses crail.vectorstore's on-disk
persistence (in-memory FAISS only), so this never touches the app's live
index/queue/audit log in data/ and logs/.

Usage:
    .venv\\Scripts\\python.exe evaluate_real_docs.py
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.vectorstores import FAISS
from openai import OpenAI

from config import OPENAI_API_KEY, TIER2_THRESHOLD, TIER3_THRESHOLD
from crail.generation import generate_answer
from crail.ingestion import ingest_pdf
from crail.retrieval import retrieve
from crail.rri import crail_decision
from crail.vectorstore import get_embeddings

client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = get_embeddings()

SOURCE_ROOT = Path(r"C:\Users\shahe\genai-chatbot\crail\data")

CURRENT_DOC = ("pair1_doc_b_nist_genai_profile.pdf", SOURCE_ROOT / "pair1" / "doc_b.pdf")

NOISE_DOCS = [
    ("pair2_doc_a_neonatal_health.pdf", SOURCE_ROOT / "pair2_neonatal" / "doc_a.pdf"),
    ("pair2_doc_b_ai_nursing_review.pdf", SOURCE_ROOT / "pair2_neonatal" / "doc_b.pdf"),
    ("pair5_doc_a_lewis_rag.pdf", SOURCE_ROOT / "pair5" / "doc_a.pdf"),
    ("pair5_doc_b_gao_rag_survey.pdf", SOURCE_ROOT / "pair5" / "doc_b.pdf"),
    ("pair6_doc_a_gpt3.pdf", SOURCE_ROOT / "pair6" / "doc_a.pdf"),
    ("pair6_doc_b_gpt4.pdf", SOURCE_ROOT / "pair6" / "doc_b.pdf"),
]

# Genuinely on-topic questions about the NIST GenAI Risk Profile document,
# specific enough that only that document should ever answer them, and
# with no topical connection to neonatal health, nursing AI applications,
# RAG retrieval methods, or GPT model architecture/capabilities.
QUERIES = [
    "What does the NIST GenAI profile say about CBRN information risks?",
    "What is confabulation according to this NIST document?",
    "What does the document say about dangerous, violent, or hateful content risks?",
    "What data privacy risks does the document associate with GAI systems?",
    "What environmental impacts does the document attribute to GAI systems?",
]


@dataclass
class QueryResult:
    scenario: str  # "isolated" or "shared"
    query: str
    contamination_ratio: float
    confidence_mismatch: bool
    rri: float
    tier: int
    answer: str
    retrieved_sources: list = field(default_factory=list)


def run_query(vs, query, scenario) -> QueryResult:
    retrieved = retrieve(vs, query, k=4)
    answer = generate_answer(query, retrieved, client)
    decision = crail_decision(retrieved, answer, CURRENT_DOC[0], confidence_method="llm_judge", client=client)
    return QueryResult(
        scenario=scenario,
        query=query,
        contamination_ratio=decision["contamination_ratio"],
        confidence_mismatch=decision["confidence_mismatch"],
        rri=decision["rri"],
        tier=decision["tier"],
        answer=answer,
        retrieved_sources=[d.metadata.get("source_doc") for d in retrieved],
    )


def load_and_chunk(doc_id: str, path: Path):
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    t0 = time.time()
    chunks = ingest_pdf(pdf_bytes, doc_id)
    print(f"  {doc_id}: {len(chunks)} chunks ({time.time()-t0:.1f}s to extract/chunk)")
    return chunks


def build_index(chunk_lists: list) -> FAISS:
    vs = None
    for chunks in chunk_lists:
        t0 = time.time()
        if vs is None:
            vs = FAISS.from_documents(chunks, embeddings)
        else:
            vs.add_documents(chunks)
        print(f"    embedded {len(chunks)} chunks in {time.time()-t0:.1f}s")
    return vs


def main():
    print(f"Thresholds in effect: TIER2={TIER2_THRESHOLD}, TIER3={TIER3_THRESHOLD}\n")

    print("=== Loading and chunking documents ===")
    current_chunks = load_and_chunk(*CURRENT_DOC)
    noise_chunk_lists = [load_and_chunk(doc_id, path) for doc_id, path in NOISE_DOCS]
    total_noise_chunks = sum(len(c) for c in noise_chunk_lists)
    total_chunks = len(current_chunks) + total_noise_chunks
    print(f"\nCurrent doc: {len(current_chunks)} chunks")
    print(f"Noise docs (3 pairs, 6 files): {total_noise_chunks} chunks")
    print(f"Total shared-index size: {total_chunks} chunks (TOP_K=4, fetch_k=20)\n")

    all_results: list[QueryResult] = []

    print("=== Building isolated index (current doc alone) ===")
    vs_isolated = build_index([current_chunks])
    print("\n=== Isolated baseline queries ===")
    for q in QUERIES:
        r = run_query(vs_isolated, q, "isolated")
        all_results.append(r)
        print(f"  '{q[:55]}...' -> contam={r.contamination_ratio} tier={r.tier}")

    print("\n=== Building shared index (current doc + 3 unrelated real pairs) ===")
    vs_shared = build_index([current_chunks] + noise_chunk_lists)
    print("\n=== Shared-index queries (same genuinely on-topic questions) ===")
    for q in QUERIES:
        r = run_query(vs_shared, q, "shared")
        all_results.append(r)
        other = sorted(set(r.retrieved_sources) - {CURRENT_DOC[0]})
        print(f"  '{q[:55]}...' -> contam={r.contamination_ratio} tier={r.tier} other_sources={other or '-'}")

    print("\n\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"\nTotal shared-index size: {total_chunks} chunks vs fetch_k=20 "
          f"(ratio {20/total_chunks:.3f}), vastly larger than the earlier synthetic test's 5-6 chunks.\n")

    isolated = [r for r in all_results if r.scenario == "isolated"]
    shared = [r for r in all_results if r.scenario == "shared"]

    i_tiers = [r.tier for r in isolated]
    s_tiers = [r.tier for r in shared]
    print(f"Isolated baseline tiers: {i_tiers}")
    print(f"Shared-index tiers:      {s_tiers}")

    held = sum(1 for r in shared if r.tier > 1)
    print(f"\nOn-topic queries held for review despite being genuinely about the "
          f"current document and unrelated to all other loaded documents: {held}/{len(shared)}")

    if held == 0:
        print("-> Zero false holds. The earlier near-universal Tier 2 flagging was a")
        print("   small-test-corpus artifact, not a property of the zero-tolerance fix itself.")
    else:
        print("-> Some on-topic queries were still held even with a large, real, diverse")
        print("   corpus. This means the zero-tolerance threshold's review-queue cost is")
        print("   not purely a small-corpus artifact.")

    with open("eval_results_real_docs.json", "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in all_results], f, indent=2)
    print("\nRaw results written to eval_results_real_docs.json")


if __name__ == "__main__":
    main()
