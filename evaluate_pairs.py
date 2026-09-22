"""Paired-document evaluation harness (the original 3-pair test).

This is the exact scenario that first exposed the Tier 2 blind spot: Pair 1
(HR handbook vs. API reference, low semantic overlap) landed at precisely
25% contamination on every query and was silently delivered as Tier 1 under
the old TIER2_THRESHOLD=0.3 / confidence floor=0.25. Kept as its own script
(separate from evaluate.py's larger 20-document growth experiment) because
it's the direct before/after regression check for that specific fix:
config.py now sets TIER2_THRESHOLD=0.15, derived from
CONTAMINATION_WEIGHT * (1/TOP_K), so this exact case should now intercept.

Uses the real pipeline throughout (real PDF extraction, real embeddings,
real generation, real RRI scoring) and deliberately bypasses
crail.vectorstore's on-disk persistence (in-memory FAISS only), so this
never touches the app's live index/queue/audit log in data/ and logs/.

Usage:
    .venv\\Scripts\\python.exe evaluate_pairs.py
"""
import json
import statistics
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


PAIRS = [
    {
        "name": "Pair 1: unrelated domains (HR handbook vs API reference)",
        "doc_a_id": "employee_benefits_handbook_2024.pdf",
        "doc_a_text": (
            "Acme Corp Employee Benefits Handbook 2024\n\n"
            "Health insurance: Acme Corp covers 90 percent of premiums for the "
            "PPO health plan for all full-time employees, effective from the "
            "first day of employment. Dependents may be added during open "
            "enrollment in November. Dental and vision coverage are bundled "
            "at no additional cost, and employees may opt into a "
            "flexible spending account with a maximum annual contribution "
            "of $3,050 for healthcare expenses and $5,000 for dependent care.\n\n"
            "Paid time off: Full-time employees accrue 18 days of paid time "
            "off per year, plus 10 paid holidays. Unused PTO up to 5 days may "
            "be carried over into the next calendar year. Part-time employees "
            "working at least 20 hours per week accrue PTO on a prorated "
            "basis according to their scheduled hours.\n\n"
            "Retirement: Acme Corp matches 401k contributions dollar-for-dollar "
            "up to 4 percent of base salary. Employees are fully vested in "
            "the match after two years of service. Enrollment is automatic "
            "at a 3 percent contribution rate unless the employee opts out "
            "or changes their contribution percentage through the benefits "
            "portal.\n\n"
            "Remote work: Employees may work remotely up to 3 days per week "
            "with manager approval. Fully remote arrangements require VP "
            "sign-off and are reviewed annually. Employees working remotely "
            "internationally must coordinate with the People Operations team "
            "at least 60 days in advance for tax and compliance reasons.\n\n"
            "Parental leave: Acme Corp offers 16 weeks of fully paid parental "
            "leave for the primary caregiver and 8 weeks for the secondary "
            "caregiver, usable within 12 months of the birth or adoption. "
            "Leave may be taken continuously or split into two blocks with "
            "manager approval.\n\n"
            "Wellness stipend: Employees receive an annual wellness stipend "
            "of $600, reimbursable for gym memberships, fitness equipment, "
            "or mental health app subscriptions, submitted through the "
            "expense system.\n\n"
            "Tuition reimbursement: Acme Corp reimburses up to $5,250 per "
            "calendar year for job-related coursework or certifications, "
            "provided the employee remains with the company for at least "
            "one year after course completion.\n\n"
            "Contact: Questions about any benefit described in this "
            "handbook should be directed to peopleops@acmecorp.example or "
            "the internal HR portal."
        ),
        "doc_b_id": "cloud_api_reference_v3.pdf",
        "doc_b_text": (
            "Acme Cloud API Reference v3\n\n"
            "Authentication: The API uses OAuth 2.0 bearer tokens. Obtain a "
            "token from POST /oauth/token using your client_id and "
            "client_secret. Tokens expire after 3600 seconds and must be "
            "refreshed using the refresh_token grant type before expiry to "
            "avoid interrupting long-running integrations.\n\n"
            "Rate limits: Standard tier accounts are limited to 600 requests "
            "per minute per API key. Exceeding this returns HTTP 429 with a "
            "Retry-After header indicating the wait time in seconds. "
            "Enterprise tier accounts may request an increase to 5000 "
            "requests per minute through the account management console.\n\n"
            "Endpoint - create order: POST /v3/orders creates a new order. "
            "Required fields are customer_id, line_items, and "
            "shipping_address. Returns the created order object with status "
            "'pending'. Optional fields include a discount_code and "
            "gift_message.\n\n"
            "Endpoint - list users: GET /v3/users returns a paginated list "
            "of users for the authenticated account. Supports a 'limit' "
            "query parameter, maximum 100, default 20, and a 'cursor' "
            "parameter for pagination beyond the first page.\n\n"
            "Endpoint - webhooks: POST /v3/webhooks registers a callback "
            "URL that receives order.created, order.shipped, and "
            "order.cancelled events. Webhook payloads are signed with "
            "HMAC-SHA256 using your webhook secret.\n\n"
            "Error codes: 400 indicates a malformed request body. 401 "
            "indicates a missing or expired token. 404 indicates the "
            "requested resource does not exist. 429 indicates the rate "
            "limit has been exceeded. 500 indicates an internal server "
            "error and the request should be retried with exponential "
            "backoff.\n\n"
            "SDKs: Official client libraries are available for Python, "
            "Node.js, and Go, all published under the acme-cloud "
            "organization on their respective package registries.\n\n"
            "Support: Enterprise customers can reach API support at "
            "api-support@acmecorp.example with a target response time of "
            "4 business hours."
        ),
        "queries": [
            "What is the API's rate limit per minute?",
            "What authentication method does the API use?",
            "What does HTTP error code 429 mean?",
            "What fields are required to create an order?",
            "What is the default and maximum limit for listing users?",
        ],
    },
    {
        "name": "Pair 2: same-template financial reports (2023 vs 2024, high overlap)",
        "doc_a_id": "acme_q4_2023_financial_report.pdf",
        "doc_a_text": (
            "Acme Corp Q4 2023 Financial Report\n\n"
            "For the fourth quarter of fiscal year 2023, Acme Corp reported "
            "total revenue of $42.3 million, an increase of 8 percent "
            "year-over-year. Net profit for the quarter was $6.1 million, "
            "representing a net margin of 14.4 percent. Revenue by segment "
            "was $27.1 million from enterprise subscriptions, $9.8 million "
            "from professional services, and $5.4 million from marketplace "
            "transaction fees.\n\n"
            "Headcount at the end of Q4 2023 stood at 340 full-time "
            "employees across all divisions, up from 310 at the end of Q4 "
            "2022. Engineering accounted for 140 of those employees, sales "
            "and marketing for 95, and general and administrative "
            "functions for the remainder.\n\n"
            "Gross margin for the quarter was 58 percent, consistent with "
            "the prior quarter. Operating expenses totaled $18.2 million, "
            "primarily driven by sales and marketing spend of $9.6 million "
            "and research and development spend of $6.1 million.\n\n"
            "Cash and cash equivalents at quarter end were $64 million, "
            "with no outstanding debt on the balance sheet. Free cash flow "
            "for the quarter was $4.9 million, positive for the sixth "
            "consecutive quarter.\n\n"
            "The board of directors approved a share buyback program of up "
            "to $10 million for fiscal year 2024, to be executed over the "
            "following four quarters, subject to market conditions.\n\n"
            "Customer metrics: Acme Corp ended the quarter with 1,240 "
            "enterprise customers, a net revenue retention rate of 112 "
            "percent, and an annualized churn rate of 6 percent.\n\n"
            "Outlook: Management guided fiscal year 2024 revenue growth of "
            "20 to 25 percent, with continued investment in the enterprise "
            "sales team and international expansion into EMEA."
        ),
        "doc_b_id": "acme_q4_2024_financial_report.pdf",
        "doc_b_text": (
            "Acme Corp Q4 2024 Financial Report\n\n"
            "For the fourth quarter of fiscal year 2024, Acme Corp reported "
            "total revenue of $57.6 million, an increase of 36 percent "
            "year-over-year. Net profit for the quarter was $9.8 million, "
            "representing a net margin of 17.0 percent. Revenue by segment "
            "was $38.9 million from enterprise subscriptions, $11.2 million "
            "from professional services, and $7.5 million from marketplace "
            "transaction fees.\n\n"
            "Headcount at the end of Q4 2024 stood at 410 full-time "
            "employees across all divisions, up from 340 at the end of Q4 "
            "2023. Engineering accounted for 175 of those employees, sales "
            "and marketing for 120, and general and administrative "
            "functions for the remainder.\n\n"
            "Gross margin for the quarter was 61 percent, an improvement "
            "over the prior year. Operating expenses totaled $22.9 million, "
            "primarily driven by continued sales and marketing investment "
            "of $12.4 million and research and development spend of $7.8 "
            "million.\n\n"
            "Cash and cash equivalents at quarter end were $91 million, "
            "with no outstanding debt on the balance sheet. Free cash flow "
            "for the quarter was $8.2 million, positive for the tenth "
            "consecutive quarter.\n\n"
            "The board of directors approved a new share buyback program of "
            "up to $15 million for fiscal year 2025, to be executed over "
            "the following four quarters, subject to market conditions.\n\n"
            "Customer metrics: Acme Corp ended the quarter with 1,680 "
            "enterprise customers, a net revenue retention rate of 118 "
            "percent, and an annualized churn rate of 4 percent.\n\n"
            "Outlook: Management guided fiscal year 2025 revenue growth of "
            "25 to 30 percent, with continued investment in the enterprise "
            "sales team and international expansion into APAC."
        ),
        "queries": [
            "What was the total revenue for the quarter?",
            "What was the net profit for the quarter?",
            "How many full-time employees does the company have?",
            "What was the gross margin for the quarter?",
            "How large is the approved share buyback program?",
        ],
    },
    {
        "name": "Pair 3: same-template product manuals (v1 vs Pro, high overlap)",
        "doc_a_id": "toaster_3000_manual_t3_2023.pdf",
        "doc_a_text": (
            "Toaster 3000 User Manual - Model T3-2023\n\n"
            "Power: The Toaster 3000 (Model T3-2023) operates at 800 watts "
            "on a standard 120V household circuit. Power consumption in "
            "standby mode is under 0.5 watts, meeting Energy Star idle "
            "power requirements.\n\n"
            "Capacity: This model has 4 toasting slots, each accommodating "
            "standard-width bread slices up to 1.2 inches thick. Slots are "
            "arranged in two independently controlled pairs.\n\n"
            "Temperature: The maximum toasting temperature is 230 degrees "
            "Celsius, with 6 selectable browning levels controlled by the "
            "dial on the front panel.\n\n"
            "Warranty: Model T3-2023 is covered by a 1-year limited "
            "warranty from the date of purchase, covering manufacturing "
            "defects only. Proof of purchase is required for warranty "
            "claims.\n\n"
            "Safety: Always unplug the unit before cleaning. Do not "
            "immerse in water. The exterior housing may become hot during "
            "use; allow the unit to cool for at least 10 minutes before "
            "storage.\n\n"
            "Cleaning: The crumb tray slides out from the base and should "
            "be emptied after every 5 uses. Wipe the exterior with a damp "
            "cloth only; do not use abrasive cleaners.\n\n"
            "Included accessories: The box includes one bread rack "
            "attachment and a printed quick-start guide. A recipe booklet "
            "is available for download from the Acme support site.\n\n"
            "Troubleshooting: If the toaster does not power on, check that "
            "it is fully seated in the outlet and that the household "
            "circuit breaker has not tripped. If slices toast unevenly, "
            "confirm both slot pairs are set to the same browning level."
        ),
        "doc_b_id": "toaster_3000_pro_manual_t3p_2024.pdf",
        "doc_b_text": (
            "Toaster 3000 Pro User Manual - Model T3P-2024\n\n"
            "Power: The Toaster 3000 Pro (Model T3P-2024) operates at 1200 "
            "watts on a standard 120V household circuit. Power consumption "
            "in standby mode is under 0.3 watts, meeting Energy Star idle "
            "power requirements.\n\n"
            "Capacity: This model has 6 toasting slots, each accommodating "
            "standard or wide-width bread slices up to 1.5 inches thick. "
            "Slots are arranged in three independently controlled pairs.\n\n"
            "Temperature: The maximum toasting temperature is 250 degrees "
            "Celsius, with 8 selectable browning levels, including a "
            "dedicated bagel setting that toasts one side only, controlled "
            "from the digital front panel.\n\n"
            "Warranty: Model T3P-2024 is covered by a 2-year limited "
            "warranty from the date of purchase, covering manufacturing "
            "defects and normal wear on heating elements. Proof of "
            "purchase is required for warranty claims.\n\n"
            "Safety: Always unplug the unit before cleaning. Do not "
            "immerse in water. The exterior housing is cool-touch rated "
            "but the slots remain hot during use; allow the unit to cool "
            "for at least 10 minutes before storage.\n\n"
            "Cleaning: The crumb tray slides out from the base and should "
            "be emptied after every 5 uses. Wipe the exterior with a damp "
            "cloth only; do not use abrasive cleaners.\n\n"
            "Included accessories: The box includes one bread rack "
            "attachment, a bagel rack attachment, and a printed "
            "quick-start guide. A recipe booklet is available for "
            "download from the Acme support site.\n\n"
            "Troubleshooting: If the toaster does not power on, check that "
            "it is fully seated in the outlet and that the household "
            "circuit breaker has not tripped. If slices toast unevenly, "
            "confirm all three slot pairs are set to the same browning "
            "level."
        ),
        "queries": [
            "How many watts does the toaster use?",
            "How many toasting slots does it have?",
            "What is the maximum toasting temperature?",
            "How long is the warranty period?",
            "Does this model have a bagel setting?",
        ],
    },
]


@dataclass
class QueryResult:
    pair: str
    scenario: str  # "clean" or "contaminated"
    query: str
    current_doc_id: str
    contamination_ratio: float
    confidence_mismatch_keyword: bool
    confidence_mismatch_llm: bool
    rri_keyword: float
    rri_llm: float
    tier_keyword: int
    tier_llm: int
    answer: str
    retrieved_sources: list = field(default_factory=list)


def run_query(vs, query, current_doc_id, pair_name, scenario) -> QueryResult:
    retrieved = retrieve(vs, query, k=4)
    answer = generate_answer(query, retrieved, client)

    d_keyword = crail_decision(retrieved, answer, current_doc_id, confidence_method="keyword", client=client)
    mismatch_llm, _ = check_confidence_mismatch(
        answer, d_keyword["contamination_ratio"], method="llm_judge", client=client
    )
    from config import CONFIDENCE_WEIGHT, CONTAMINATION_WEIGHT

    rri_llm = min(
        d_keyword["contamination_ratio"] * CONTAMINATION_WEIGHT + (CONFIDENCE_WEIGHT if mismatch_llm else 0.0), 1.0
    )
    tier_llm = 3 if rri_llm >= TIER3_THRESHOLD else (2 if rri_llm >= TIER2_THRESHOLD else 1)

    return QueryResult(
        pair=pair_name,
        scenario=scenario,
        query=query,
        current_doc_id=current_doc_id,
        contamination_ratio=d_keyword["contamination_ratio"],
        confidence_mismatch_keyword=d_keyword["confidence_mismatch"],
        confidence_mismatch_llm=mismatch_llm,
        rri_keyword=d_keyword["rri"],
        rri_llm=round(rri_llm, 3),
        tier_keyword=d_keyword["tier"],
        tier_llm=tier_llm,
        answer=answer,
        retrieved_sources=[d.metadata.get("source_doc") for d in retrieved],
    )


def main():
    print(f"Thresholds in effect: TIER2={TIER2_THRESHOLD}, TIER3={TIER3_THRESHOLD}\n")
    all_results: list[QueryResult] = []

    for pair in PAIRS:
        print(f"\n=== {pair['name']} ===")

        docs_a = ingest_pdf(make_pdf_bytes(pair["doc_a_id"], pair["doc_a_text"]), pair["doc_a_id"])
        docs_b = ingest_pdf(make_pdf_bytes(pair["doc_b_id"], pair["doc_b_text"]), pair["doc_b_id"])
        print(f"doc A -> {len(docs_a)} chunks, doc B -> {len(docs_b)} chunks")

        vs_clean = FAISS.from_documents(docs_b, embeddings)
        for q in pair["queries"]:
            r = run_query(vs_clean, q, pair["doc_b_id"], pair["name"], "clean")
            all_results.append(r)
            print(f"  [clean] '{q[:50]}...' -> contam={r.contamination_ratio} tier(kw)={r.tier_keyword}")

        vs_contaminated = FAISS.from_documents(docs_a, embeddings)
        vs_contaminated.add_documents(docs_b)
        for q in pair["queries"]:
            r = run_query(vs_contaminated, q, pair["doc_b_id"], pair["name"], "contaminated")
            all_results.append(r)
            print(
                f"  [contam] '{q[:50]}...' -> contam={r.contamination_ratio} "
                f"tier(kw)={r.tier_keyword} tier(llm)={r.tier_llm} sources={r.retrieved_sources}"
            )

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for scenario in ["clean", "contaminated"]:
        subset = [r for r in all_results if r.scenario == scenario]
        n = len(subset)
        tier_counts = {1: 0, 2: 0, 3: 0}
        for r in subset:
            tier_counts[r.tier_keyword] += 1
        intercepted = tier_counts[2] + tier_counts[3]
        avg_contam = statistics.mean(r.contamination_ratio for r in subset) if subset else 0
        print(f"\n[{scenario.upper()}] n={n}")
        print(f"  avg contamination ratio: {avg_contam:.3f}")
        print(f"  tier distribution (keyword method): {tier_counts}")
        print(f"  interception rate: {intercepted}/{n} = {intercepted/n:.0%}" if n else "  n/a")

    contam = [r for r in all_results if r.scenario == "contaminated"]
    agree = sum(1 for r in contam if r.tier_keyword == r.tier_llm)
    print(f"\nKeyword vs LLM-judge tier agreement (contaminated cases): {agree}/{len(contam)}")

    print("\nPer-pair-type breakdown (contaminated scenario):")
    for pair in PAIRS:
        subset = [r for r in contam if r.pair == pair["name"]]
        avg_contam = statistics.mean(r.contamination_ratio for r in subset)
        intercepted_kw = sum(1 for r in subset if r.tier_keyword > 1)
        intercepted_llm = sum(1 for r in subset if r.tier_llm > 1)
        print(f"  {pair['name']}")
        print(f"    avg contamination: {avg_contam:.3f} | intercepted (keyword): {intercepted_kw}/{len(subset)} "
              f"| intercepted (llm judge): {intercepted_llm}/{len(subset)}")

    boundary_cases = [r for r in contam if abs(r.contamination_ratio - 0.25) < 1e-9]
    print(f"\nExactly-25%-contamination cases observed naturally: {len(boundary_cases)}")
    for r in boundary_cases:
        print(f"  '{r.query[:50]}' tier(kw)={r.tier_keyword} rri(kw)={r.rri_keyword} "
              f"confidence_mismatch={r.confidence_mismatch_keyword}")

    clean = [r for r in all_results if r.scenario == "clean"]
    false_positives = [r for r in clean if r.tier_keyword != 1]
    print(f"\nFalse positives in clean scenario (should be 0): {len(false_positives)}")
    for r in false_positives:
        print(f"  '{r.query[:50]}' contam={r.contamination_ratio} tier={r.tier_keyword}")

    with open("eval_results_pairs.json", "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in all_results], f, indent=2)
    print("\nRaw results written to eval_results_pairs.json")


if __name__ == "__main__":
    main()
