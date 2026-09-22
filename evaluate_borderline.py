"""Borderline evaluation: can the new zero-tolerance threshold tell the
difference between harmful and harmless contamination? (Spoiler, by
design, it structurally cannot on its own - that's what this test is
actually measuring.)

Motivation: the pairwise and growth-experiment harnesses only tested two
extremes - fully isolated documents (0% contamination, trivially Tier 1
under any sane threshold) and deliberately stale/contradictory document
pairs (old vs new report, clearly should be caught). Neither tells us
whether TIER2_THRESHOLD=0.15 (any single wrong chunk out of four) creates
excessive Tier 2 volume on genuinely benign cross-document overlap, the
realistic middle case most production multi-document indexes actually
contain.

Design: one "current" document (an employee handbook) whose remote-work
section is deliberately echoed, consistently, in a separate standalone
remote-work policy document (doc_b, benign overlap: same fact, different
source). A third document is a superseded, contradictory version of that
same policy (doc_c, genuine staleness: different fact, different source).
Both doc_b and doc_c sit in the same shared index as the handbook. The
same query ("how many days can I work remotely") is run repeatedly; which
of doc_b/doc_c's chunk gets pulled in by retrieval varies query to query
depending on chance/ranking, letting us compare CRAIL's behavior (and the
actual generated answer's correctness) across both leak types under
identical RRI mechanics.

Usage:
    .venv\\Scripts\\python.exe evaluate_borderline.py
"""
import json
from dataclasses import dataclass, field

from fpdf import FPDF
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from config import OPENAI_API_KEY, TIER2_THRESHOLD, TIER3_THRESHOLD
from crail.confidence import check_confidence_mismatch
from crail.generation import generate_answer
from crail.ingestion import ingest_pdf
from crail.retrieval import retrieve
from crail.rri import crail_decision
from crail.vectorstore import get_embeddings

client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = get_embeddings()


def make_pdf_bytes(title: str, body: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, title)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(4)
    for line in body.split("\n"):
        if line.strip():
            pdf.multi_cell(0, 6, line)
        else:
            pdf.ln(3)
    return bytes(pdf.output())


CURRENT_DOC_ID = "acme_employee_handbook.pdf"
CURRENT_DOC_TEXT = (
    "Acme Corp Employee Handbook\n\n"
    "Paid time off: Full-time employees accrue 18 days of paid time off "
    "per year, plus 10 paid holidays. Unused PTO up to 5 days may be "
    "carried over into the next calendar year. Part-time employees "
    "working at least 20 hours per week accrue PTO on a prorated basis.\n\n"
    "Health benefits: Acme Corp covers 90 percent of premiums for the PPO "
    "health plan for all full-time employees, effective from the first "
    "day of employment. Dental and vision coverage are bundled at no "
    "additional cost. Employees may opt into a flexible spending account "
    "with a maximum annual contribution of $3,050.\n\n"
    "Retirement: Acme Corp matches 401k contributions dollar-for-dollar "
    "up to 4 percent of base salary. Employees are fully vested in the "
    "match after two years of service.\n\n"
    "Remote work: Employees may work remotely up to 2 days per week, "
    "consistent with the company Remote Work Policy. Requests for "
    "additional remote days require manager approval and are reviewed "
    "on a case-by-case basis by the employee's department head.\n\n"
    "Expense reimbursement: Business expenses are reimbursed within 10 "
    "business days of submission through the expense portal, provided "
    "receipts are attached for any expense over $25. Travel expenses "
    "require pre-approval for trips exceeding $500 in total cost.\n\n"
    "Code of conduct: Employees are expected to treat colleagues, "
    "clients, and vendors with respect. Harassment, discrimination, and "
    "retaliation are strictly prohibited. Violations are handled by "
    "Human Resources on a case-by-case basis and may result in "
    "disciplinary action up to and including termination.\n\n"
    "Onboarding: New hires complete a 2-week onboarding program covering "
    "company systems, security training, and role-specific shadowing "
    "with a designated mentor.\n\n"
    "Performance reviews: Formal performance reviews are conducted twice "
    "annually, in June and December, with informal check-ins encouraged "
    "monthly between employees and managers.\n\n"
    "Parental leave: Acme Corp offers 16 weeks of fully paid parental "
    "leave for the primary caregiver and 8 weeks for the secondary "
    "caregiver, usable within 12 months of the birth or adoption.\n\n"
    "Workplace safety: Employees must report any workplace safety "
    "concerns to their manager or the Facilities team within 24 hours "
    "of observing the issue."
)

# Benign overlap: a separate, standalone document that legitimately covers
# the same policy and agrees with the handbook. Not stale, not wrong, just
# a different document that happens to state the same fact.
CONSISTENT_DOC_ID = "acme_remote_work_policy_current.pdf"
CONSISTENT_DOC_TEXT = (
    "Acme Corp Remote Work Policy (Current)\n\n"
    "Purpose: This policy defines the terms under which Acme Corp "
    "employees may work outside of a company office location.\n\n"
    "Eligibility: All full-time employees who have completed their 90-day "
    "introductory period are eligible to work remotely, subject to role "
    "requirements and manager discretion. Certain roles requiring "
    "physical presence, such as facilities and on-site support, are "
    "excluded from this policy.\n\n"
    "Remote days: Employees may work remotely up to 2 days per week. "
    "Additional remote days beyond this require written manager approval "
    "and are reviewed quarterly by the department head.\n\n"
    "Equipment: Acme Corp provides a laptop and a one-time $200 home "
    "office stipend for remote-eligible employees, reimbursable through "
    "the standard expense process within the first 90 days of "
    "eligibility.\n\n"
    "Security: Remote employees must connect to internal systems through "
    "the company VPN and may not access company data on personal "
    "devices. Multi-factor authentication is required for all remote "
    "logins.\n\n"
    "Core hours: Remote employees are expected to be available during "
    "core collaboration hours of 10am to 3pm in their local time zone, "
    "regardless of their broader working schedule.\n\n"
    "International remote work: Employees working remotely from outside "
    "their home country for more than 2 weeks must notify People "
    "Operations in advance for tax and compliance reasons. Extended "
    "international remote work beyond 90 days requires additional "
    "approval from Legal.\n\n"
    "Office visits: Remote-eligible employees are expected to visit their "
    "home office location at least once per quarter for team events, "
    "at company expense if travel is required.\n\n"
    "Policy review: This policy is reviewed annually by People "
    "Operations and may be updated to reflect changing business needs."
)

# Genuine staleness: a superseded version of the SAME policy with a
# DIFFERENT, contradictory fact (1 day instead of 2). This is the real
# Phantom Context Hallucination case, if this leaks in instead, the
# answer could actually become factually wrong.
STALE_DOC_ID = "acme_remote_work_policy_2022_superseded.pdf"
STALE_DOC_TEXT = (
    "Acme Corp Remote Work Policy (2022, superseded)\n\n"
    "Purpose: This policy defines the terms under which Acme Corp "
    "employees may work outside of a company office location.\n\n"
    "Eligibility: All full-time employees who have completed their 90-day "
    "introductory period are eligible to work remotely, subject to role "
    "requirements and manager discretion. Certain roles requiring "
    "physical presence, such as facilities and on-site support, are "
    "excluded from this policy.\n\n"
    "Remote days: Employees may work remotely up to 1 day per week, "
    "typically Wednesday or Friday, subject to team coverage "
    "requirements. Exceptions require director-level approval.\n\n"
    "Equipment: Acme Corp provides a laptop for remote-eligible "
    "employees. No home office stipend is provided under this policy.\n\n"
    "Security: Remote employees must connect to internal systems through "
    "the company VPN. Multi-factor authentication is required for all "
    "remote logins.\n\n"
    "Core hours: Remote employees are expected to be available during "
    "core collaboration hours of 10am to 3pm in their local time zone.\n\n"
    "International remote work: Employees working remotely from outside "
    "their home country are not permitted under this policy without "
    "prior written approval from both People Operations and Legal.\n\n"
    "Office visits: Remote-eligible employees are expected to visit their "
    "home office location at least twice per month for team events.\n\n"
    "Amendment history: This policy was last amended in March 2022. It "
    "was superseded by the current Remote Work Policy effective January "
    "2023, which changed the maximum remote days and added a home office "
    "stipend."
)

# Queries entirely unrelated to remote work, to confirm the sanity
# baseline still holds (should stay clean regardless of what else is in
# the shared index).
BASELINE_QUERIES = [
    "How many paid time off days do employees get per year?",
    "What percentage of health insurance premiums does the company cover?",
    "How many business days does expense reimbursement take?",
]

# The borderline-trigger query, run multiple times to sample retrieval's
# natural variation in which "other" document's chunk gets pulled in.
REMOTE_WORK_QUERY = "How many days per week can employees work remotely?"
REMOTE_WORK_QUERY_TRIALS = 8


@dataclass
class QueryResult:
    condition: str  # "baseline", "vs_consistent_doc", "vs_stale_doc"
    query: str
    contamination_ratio: float
    confidence_mismatch_llm: bool
    rri: float
    tier: int
    answer: str
    answer_says_2_days: bool
    answer_says_1_day: bool
    retrieved_sources: list = field(default_factory=list)


def run_query(vs, query, condition) -> QueryResult:
    retrieved = retrieve(vs, query, k=4)
    answer = generate_answer(query, retrieved, client)
    decision = crail_decision(retrieved, answer, CURRENT_DOC_ID, confidence_method="llm_judge", client=client)

    return QueryResult(
        condition=condition,
        query=query,
        contamination_ratio=decision["contamination_ratio"],
        confidence_mismatch_llm=decision["confidence_mismatch"],
        rri=decision["rri"],
        tier=decision["tier"],
        answer=answer,
        answer_says_2_days="2 day" in answer.lower() or "two day" in answer.lower(),
        answer_says_1_day="1 day" in answer.lower() or "one day" in answer.lower(),
        retrieved_sources=[d.metadata.get("source_doc") for d in retrieved],
    )


def main():
    print(f"Thresholds in effect: TIER2={TIER2_THRESHOLD}, TIER3={TIER3_THRESHOLD}\n")

    handbook_chunks = ingest_pdf(make_pdf_bytes(CURRENT_DOC_ID, CURRENT_DOC_TEXT), CURRENT_DOC_ID)
    consistent_chunks = ingest_pdf(make_pdf_bytes(CONSISTENT_DOC_ID, CONSISTENT_DOC_TEXT), CONSISTENT_DOC_ID)
    stale_chunks = ingest_pdf(make_pdf_bytes(STALE_DOC_ID, STALE_DOC_TEXT), STALE_DOC_ID)
    print(f"handbook -> {len(handbook_chunks)} chunks")
    print(f"consistent policy doc -> {len(consistent_chunks)} chunks")
    print(f"stale policy doc -> {len(stale_chunks)} chunks\n")

    all_results: list[QueryResult] = []

    # --- Condition A: handbook alone (isolated baseline) ---
    print("=== Condition A: handbook alone (isolated baseline) ===")
    vs_alone = FAISS.from_documents(handbook_chunks, embeddings)
    for q in BASELINE_QUERIES:
        r = run_query(vs_alone, q, "baseline")
        all_results.append(r)
        print(f"  '{q[:50]}...' -> contam={r.contamination_ratio} tier={r.tier}")
    r = run_query(vs_alone, REMOTE_WORK_QUERY, "baseline")
    all_results.append(r)
    print(f"  '{REMOTE_WORK_QUERY[:50]}...' -> contam={r.contamination_ratio} tier={r.tier} answer='{r.answer[:80]}'")

    # --- Condition B: handbook + shared index also containing the
    # CONSISTENT (benign, non-contradictory) remote work policy ---
    print("\n=== Condition B: handbook + benign consistent overlap doc, shared never-cleared index ===")
    vs_consistent = FAISS.from_documents(handbook_chunks, embeddings)
    vs_consistent.add_documents(consistent_chunks)
    for q in BASELINE_QUERIES:
        r = run_query(vs_consistent, q, "vs_consistent_doc")
        all_results.append(r)
        print(f"  [unrelated topic] '{q[:50]}...' -> contam={r.contamination_ratio} tier={r.tier}")
    for i in range(REMOTE_WORK_QUERY_TRIALS):
        r = run_query(vs_consistent, REMOTE_WORK_QUERY, "vs_consistent_doc")
        all_results.append(r)
        print(f"  [remote-work trial {i+1}] contam={r.contamination_ratio} tier={r.tier} "
              f"sources={r.retrieved_sources} answer='{r.answer[:80]}'")

    # --- Condition C: handbook + shared index also containing the STALE
    # (genuinely contradictory) remote work policy ---
    print("\n=== Condition C: handbook + genuinely stale contradicting doc, shared never-cleared index ===")
    vs_stale = FAISS.from_documents(handbook_chunks, embeddings)
    vs_stale.add_documents(stale_chunks)
    for q in BASELINE_QUERIES:
        r = run_query(vs_stale, q, "vs_stale_doc")
        all_results.append(r)
        print(f"  [unrelated topic] '{q[:50]}...' -> contam={r.contamination_ratio} tier={r.tier}")
    for i in range(REMOTE_WORK_QUERY_TRIALS):
        r = run_query(vs_stale, REMOTE_WORK_QUERY, "vs_stale_doc")
        all_results.append(r)
        print(f"  [remote-work trial {i+1}] contam={r.contamination_ratio} tier={r.tier} "
              f"sources={r.retrieved_sources} answer='{r.answer[:80]}'")

    # --- Report ---
    print("\n\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for condition, label in [
        ("baseline", "Isolated handbook (sanity baseline)"),
        ("vs_consistent_doc", "Handbook + BENIGN consistent overlap doc"),
        ("vs_stale_doc", "Handbook + GENUINELY STALE contradicting doc"),
    ]:
        subset = [r for r in all_results if r.condition == condition]
        remote_subset = [r for r in subset if r.query == REMOTE_WORK_QUERY]
        unrelated_subset = [r for r in subset if r.query != REMOTE_WORK_QUERY]

        print(f"\n[{label}]")
        u_tiers = [r.tier for r in unrelated_subset]
        print(f"  Unrelated-topic queries (n={len(unrelated_subset)}): tiers={u_tiers} "
              f"(expect all Tier 1, nothing here should ever mention remote work)")

        if remote_subset:
            n = len(remote_subset)
            held = sum(1 for r in remote_subset if r.tier > 1)
            correct_2day = sum(1 for r in remote_subset if r.answer_says_2_days and not r.answer_says_1_day)
            wrong_1day = sum(1 for r in remote_subset if r.answer_says_1_day and not r.answer_says_2_days)
            contam_values = [r.contamination_ratio for r in remote_subset]
            print(f"  Remote-work query (n={n}): held for review {held}/{n}, "
                  f"contamination values={contam_values}")
            print(f"    answer correctly said '2 days' (no contradiction): {correct_2day}/{n}")
            print(f"    answer incorrectly said '1 day' (adopted the stale fact): {wrong_1day}/{n}")

    print("\n--- The key comparison ---")
    consistent_remote = [r for r in all_results if r.condition == "vs_consistent_doc" and r.query == REMOTE_WORK_QUERY]
    stale_remote = [r for r in all_results if r.condition == "vs_stale_doc" and r.query == REMOTE_WORK_QUERY]
    c_held = sum(1 for r in consistent_remote if r.tier > 1)
    s_held = sum(1 for r in stale_remote if r.tier > 1)
    print(f"Benign-overlap doc: held {c_held}/{len(consistent_remote)} of the time")
    print(f"Stale/contradicting doc: held {s_held}/{len(stale_remote)} of the time")
    if c_held == s_held:
        print("-> CRAIL held BOTH at the same rate: it cannot tell benign overlap from harmful")
        print("   staleness by contamination_ratio alone. That's expected (it's not designed")
        print("   to), but it means every hold in the benign case is a real review-queue cost")
        print("   with no automatic way to skip it.")

    with open("eval_results_borderline.json", "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in all_results], f, indent=2)
    print("\nRaw results written to eval_results_borderline.json")


if __name__ == "__main__":
    main()
