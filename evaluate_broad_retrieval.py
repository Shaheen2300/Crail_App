"""Empirical end-to-end test of the summary/analysis intent path.

Verifies the actual code (not a reimplementation of the logic) does what
the math in crail/rri.py's effective_tier2_threshold() promises: 5
summary/analysis-intent queries run against an ISOLATED single-document
index each (should always be clean, Tier 1, contamination=0), and the
same 5 queries run again against a SHARED index containing all 3 real
documents together, never cleared (real contamination should be measured
correctly and never silently missed, given the scaled threshold).

Uses the real documents already live in this project's index right now
(two real resumes and a real neonatal-health research report extracted
via pypdf, real embeddings, real generation, real RRI scoring) rather
than synthetic stand-ins. Reads chunks out of the current on-disk index
read-only (via crail.viz) and builds fresh in-memory FAISS indices from
them, so this never mutates the app's live index/queue/audit log.

Usage:
    .venv\\Scripts\\python.exe evaluate_broad_retrieval.py
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from openai import OpenAI

from config import BROAD_TOP_K, OPENAI_API_KEY, TOP_K
from crail import vectorstore as crail_vectorstore
from crail.generation import generate_answer
from crail.intent import classify_intent
from crail.retrieval import retrieve, retrieve_broad
from crail.rri import crail_decision
from crail.viz import get_all_chunks_with_vectors

client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = crail_vectorstore.get_embeddings()

# 5 queries spanning both intents (summary + analysis) and all 3 real
# documents currently in the live index.
QUERIES = [
    ("summarize this document", "neonatal_report.pdf"),
    ("what are the key findings and conclusions?", "neonatal_report.pdf"),
    ("give me an overview of my resume", "Shaheen_Memon_Resume_2026..pdf"),
    ("analyze my professional experience and give key takeaways", "Shaheen_Memon_Resume_2026..pdf"),
    ("summarize my academic background and research", "PHD_Resume.pdf"),
]


@dataclass
class QueryResult:
    scenario: str  # "isolated" or "shared"
    query: str
    current_doc_id: str
    intent: str
    num_retrieved: int
    contamination_ratio: float
    tier2_threshold: float
    rri: float
    tier: int
    answer: str
    retrieved_sources: list = field(default_factory=list)


def run_query(vs, query, current_doc_id, scenario) -> QueryResult:
    intent = classify_intent(query)
    if intent == "specific":
        retrieved = retrieve(vs, query, k=TOP_K)
    else:
        retrieved = retrieve_broad(vs, query, current_doc_id, k=BROAD_TOP_K)

    answer = generate_answer(query, retrieved, client, intent=intent)
    decision = crail_decision(retrieved, answer, current_doc_id, confidence_method="llm_judge", client=client)

    return QueryResult(
        scenario=scenario,
        query=query,
        current_doc_id=current_doc_id,
        intent=intent,
        num_retrieved=len(retrieved),
        contamination_ratio=decision["contamination_ratio"],
        tier2_threshold=decision["tier2_threshold"],
        rri=decision["rri"],
        tier=decision["tier"],
        answer=answer,
        retrieved_sources=[d.metadata.get("source_doc") for d in retrieved],
    )


def main():
    print("=== Reading real chunks from the live index (read-only) ===")
    vs_live = crail_vectorstore.load_vectorstore()
    if vs_live is None:
        print("No live index found. Upload documents in the app first.")
        return

    all_docs, _ = get_all_chunks_with_vectors(vs_live)
    by_doc = defaultdict(list)
    for d in all_docs:
        by_doc[d.metadata.get("source_doc")].append(d)
    for doc_id, chunks in by_doc.items():
        print(f"  {doc_id}: {len(chunks)} chunks")

    needed_docs = {doc_id for _, doc_id in QUERIES}
    missing = needed_docs - set(by_doc.keys())
    if missing:
        print(f"Missing required documents in the live index: {missing}")
        return

    print("\n=== Building fresh in-memory indices (isolated + shared) ===")
    isolated_indices = {}
    for doc_id in needed_docs:
        isolated_indices[doc_id] = FAISS.from_documents(by_doc[doc_id], embeddings)
        print(f"  isolated index for {doc_id}: {len(by_doc[doc_id])} chunks")

    vs_shared = None
    for doc_id, chunks in by_doc.items():
        if vs_shared is None:
            vs_shared = FAISS.from_documents(chunks, embeddings)
        else:
            vs_shared.add_documents(chunks)
    print(f"  shared index: {sum(len(c) for c in by_doc.values())} total chunks across {len(by_doc)} documents")

    all_results: list[QueryResult] = []

    print("\n=== ISOLATED (clean) scenario: 5 queries ===")
    for query, doc_id in QUERIES:
        r = run_query(isolated_indices[doc_id], query, doc_id, "isolated")
        all_results.append(r)
        print(f"  '{query[:55]}' [{r.intent}] -> retrieved={r.num_retrieved} contam={r.contamination_ratio} "
              f"threshold={r.tier2_threshold} rri={r.rri} tier={r.tier}")

    print("\n=== SHARED (never-cleared, contaminated) scenario: same 5 queries ===")
    for query, doc_id in QUERIES:
        r = run_query(vs_shared, query, doc_id, "shared")
        all_results.append(r)
        other = sorted(set(r.retrieved_sources) - {doc_id})
        print(f"  '{query[:55]}' [{r.intent}] -> retrieved={r.num_retrieved} contam={r.contamination_ratio} "
              f"threshold={r.tier2_threshold} rri={r.rri} tier={r.tier} other_sources={other or '-'}")

    print("\n\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    isolated = [r for r in all_results if r.scenario == "isolated"]
    shared = [r for r in all_results if r.scenario == "shared"]

    false_positives = [r for r in isolated if r.tier != 1 or r.contamination_ratio != 0.0]
    print(f"\n[ISOLATED] n={len(isolated)}, false positives (should be 0): {len(false_positives)}")
    for r in false_positives:
        print(f"  UNEXPECTED: '{r.query}' contam={r.contamination_ratio} tier={r.tier}")
    intents_used = {r.intent for r in isolated}
    print(f"  Intents actually classified: {intents_used} (expect 'summary' and/or 'analysis', not 'specific')")
    k_used = {r.num_retrieved for r in isolated}
    print(f"  Retrieval sizes actually used: {k_used} (expect >4, confirming broad mode engaged)")

    print(f"\n[SHARED] n={len(shared)}")
    missed = [r for r in shared if r.contamination_ratio > 0 and r.tier == 1]
    caught = [r for r in shared if r.contamination_ratio > 0 and r.tier > 1]
    clean_and_clean = [r for r in shared if r.contamination_ratio == 0]
    print(f"  Genuinely contaminated queries caught (Tier > 1): {len(caught)}")
    print(f"  Genuinely contaminated queries SILENTLY MISSED (Tier 1 despite contamination > 0): {len(missed)}")
    for r in missed:
        print(f"    MISS: '{r.query}' contam={r.contamination_ratio} threshold={r.tier2_threshold} rri={r.rri}")
    print(f"  Queries with zero actual contamination this run (retrieval happened to stay clean): {len(clean_and_clean)}")

    print("\nFull per-query detail (shared scenario):")
    for r in shared:
        print(f"  '{r.query}'")
        print(f"    intent={r.intent} retrieved={r.num_retrieved} contam={r.contamination_ratio} "
              f"threshold={r.tier2_threshold} rri={r.rri} tier={r.tier}")
        print(f"    sources={r.retrieved_sources}")

    with open("eval_results_broad_retrieval.json", "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in all_results], f, indent=2)
    print("\nRaw results written to eval_results_broad_retrieval.json")


if __name__ == "__main__":
    main()
