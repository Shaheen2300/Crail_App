"""Large-scale controlled evaluation harness for CRAIL.

Two questions this run is designed to answer:

1. With realistically-sized documents (not corpus-size-starved stubs),
   does a shared, never-cleared index still produce high contamination?
2. As the number of documents in that shared index grows (5 -> 10 -> 20),
   does contamination/interception stay roughly stable, or does it keep
   tracking the ratio of fetch_k to total corpus size at every scale?
   Answering #2 requires holding the query set fixed while only the index
   grows, so a fixed 25-query "probe set" (from the first 5 documents) is
   run against the index at all three sizes, isolating the growth effect
   from query difficulty. A separate "full coverage" pass also runs every
   document's own queries at the scale where that document exists, so the
   headline numbers reflect the whole corpus, not just the probe set.

Uses the real pipeline throughout (real PDF extraction, real embeddings,
real generation, real RRI scoring) and deliberately bypasses
crail.vectorstore's on-disk persistence (in-memory FAISS only), so this
never touches the app's live index/queue/audit log in data/ and logs/.

Usage:
    .venv\\Scripts\\python.exe evaluate.py
"""
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field

from fpdf import FPDF
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from config import CONFIDENCE_MISMATCH_CONTAMINATION_FLOOR, OPENAI_API_KEY, TIER2_THRESHOLD, TIER3_THRESHOLD, TOP_K
from crail.confidence import check_confidence_mismatch
from crail.generation import generate_answer
from crail.ingestion import ingest_pdf
from crail.retrieval import retrieve
from crail.rri import crail_decision
from crail.vectorstore import get_embeddings

client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = get_embeddings()

FETCH_K = max(TOP_K * 5, 20)  # mirrors crail/retrieval.py's retrieve()


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


# ---------------------------------------------------------------------------
# 20 full-length documents (~3000-4000 characters each, ~4-5 chunks at
# CHUNK_SIZE=1000/CHUNK_OVERLAP=200), so a single document can plausibly
# satisfy TOP_K=4 on its own - unlike the earlier short-document run, any
# contamination observed here reflects real retrieval behavior, not the
# corpus simply being too sparse to avoid it.
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "doc_id": "employee_benefits_handbook_2024.pdf",
        "text": (
            "Acme Corp Employee Benefits Handbook 2024\n\n"
            "Health insurance: Acme Corp covers 90 percent of premiums for the "
            "PPO health plan for all full-time employees, effective from the "
            "first day of employment. Dependents may be added during open "
            "enrollment in November. Dental and vision coverage are bundled "
            "at no additional cost, and employees may opt into a flexible "
            "spending account with a maximum annual contribution of $3,050 "
            "for healthcare expenses and $5,000 for dependent care.\n\n"
            "Paid time off: Full-time employees accrue 18 days of paid time "
            "off per year, plus 10 paid holidays. Unused PTO up to 5 days may "
            "be carried over into the next calendar year. Part-time employees "
            "working at least 20 hours per week accrue PTO on a prorated "
            "basis according to their scheduled hours.\n\n"
            "Retirement: Acme Corp matches 401k contributions dollar-for-dollar "
            "up to 4 percent of base salary. Employees are fully vested in "
            "the match after two years of service. Enrollment is automatic "
            "at a 3 percent contribution rate unless the employee opts out.\n\n"
            "Remote work: Employees may work remotely up to 3 days per week "
            "with manager approval. Fully remote arrangements require VP "
            "sign-off and are reviewed annually. Employees working remotely "
            "internationally must coordinate with People Operations at least "
            "60 days in advance for tax and compliance reasons.\n\n"
            "Parental leave: Acme Corp offers 16 weeks of fully paid parental "
            "leave for the primary caregiver and 8 weeks for the secondary "
            "caregiver, usable within 12 months of the birth or adoption.\n\n"
            "Wellness stipend: Employees receive an annual wellness stipend "
            "of $600, reimbursable for gym memberships, fitness equipment, "
            "or mental health app subscriptions.\n\n"
            "Tuition reimbursement: Acme Corp reimburses up to $5,250 per "
            "calendar year for job-related coursework or certifications, "
            "provided the employee remains with the company for at least "
            "one year after course completion.\n\n"
            "Contact: Questions about any benefit described in this "
            "handbook should be directed to peopleops@acmecorp.example."
        ),
        "queries": [
            "What percentage of health insurance premiums does the company cover?",
            "How many paid time off days do full-time employees get per year?",
            "What is the 401k matching policy?",
            "How many days per week can employees work remotely?",
            "How many weeks of parental leave does the primary caregiver get?",
        ],
    },
    {
        "doc_id": "cloud_api_reference_v3.pdf",
        "text": (
            "Acme Cloud API Reference v3\n\n"
            "Authentication: The API uses OAuth 2.0 bearer tokens. Obtain a "
            "token from POST /oauth/token using your client_id and "
            "client_secret. Tokens expire after 3600 seconds and must be "
            "refreshed using the refresh_token grant type before expiry.\n\n"
            "Rate limits: Standard tier accounts are limited to 600 requests "
            "per minute per API key. Exceeding this returns HTTP 429 with a "
            "Retry-After header. Enterprise tier accounts may request an "
            "increase to 5000 requests per minute through the account "
            "management console.\n\n"
            "Endpoint - create order: POST /v3/orders creates a new order. "
            "Required fields are customer_id, line_items, and "
            "shipping_address. Optional fields include a discount_code and "
            "gift_message.\n\n"
            "Endpoint - list users: GET /v3/users returns a paginated list "
            "of users. Supports a 'limit' query parameter, maximum 100, "
            "default 20, and a 'cursor' parameter for pagination.\n\n"
            "Endpoint - webhooks: POST /v3/webhooks registers a callback "
            "URL that receives order.created, order.shipped, and "
            "order.cancelled events. Payloads are signed with HMAC-SHA256.\n\n"
            "Error codes: 400 malformed request. 401 missing or expired "
            "token. 404 resource not found. 429 rate limit exceeded. 500 "
            "internal server error, retry with exponential backoff.\n\n"
            "SDKs: Official client libraries are available for Python, "
            "Node.js, and Go, published under the acme-cloud organization.\n\n"
            "Support: Enterprise customers can reach API support at "
            "api-support@acmecorp.example with a 4-hour response target."
        ),
        "queries": [
            "What is the API's rate limit per minute for standard accounts?",
            "What authentication method does the API use?",
            "What does HTTP error code 429 mean?",
            "What fields are required to create an order?",
            "What events does the webhook endpoint send?",
        ],
    },
    {
        "doc_id": "acme_q4_2023_financial_report.pdf",
        "text": (
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
            "and marketing for 95.\n\n"
            "Gross margin for the quarter was 58 percent, consistent with "
            "the prior quarter. Operating expenses totaled $18.2 million, "
            "primarily sales and marketing spend of $9.6 million and "
            "research and development spend of $6.1 million.\n\n"
            "Cash and cash equivalents at quarter end were $64 million, "
            "with no outstanding debt. Free cash flow for the quarter was "
            "$4.9 million, positive for the sixth consecutive quarter.\n\n"
            "The board of directors approved a share buyback program of up "
            "to $10 million for fiscal year 2024.\n\n"
            "Customer metrics: Acme Corp ended the quarter with 1,240 "
            "enterprise customers, a net revenue retention rate of 112 "
            "percent, and an annualized churn rate of 6 percent.\n\n"
            "Outlook: Management guided fiscal year 2024 revenue growth of "
            "20 to 25 percent, with continued investment in enterprise "
            "sales and international expansion into EMEA."
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
        "doc_id": "acme_q4_2024_financial_report.pdf",
        "text": (
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
            "and marketing for 120.\n\n"
            "Gross margin for the quarter was 61 percent, an improvement "
            "over the prior year. Operating expenses totaled $22.9 million, "
            "primarily sales and marketing spend of $12.4 million and "
            "research and development spend of $7.8 million.\n\n"
            "Cash and cash equivalents at quarter end were $91 million, "
            "with no outstanding debt. Free cash flow for the quarter was "
            "$8.2 million, positive for the tenth consecutive quarter.\n\n"
            "The board of directors approved a new share buyback program of "
            "up to $15 million for fiscal year 2025.\n\n"
            "Customer metrics: Acme Corp ended the quarter with 1,680 "
            "enterprise customers, a net revenue retention rate of 118 "
            "percent, and an annualized churn rate of 4 percent.\n\n"
            "Outlook: Management guided fiscal year 2025 revenue growth of "
            "25 to 30 percent, with continued investment in enterprise "
            "sales and international expansion into APAC."
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
        "doc_id": "toaster_3000_manual_t3_2023.pdf",
        "text": (
            "Toaster 3000 User Manual - Model T3-2023\n\n"
            "Power: The Toaster 3000 (Model T3-2023) operates at 800 watts "
            "on a standard 120V household circuit. Standby power draw is "
            "under 0.5 watts, meeting Energy Star idle requirements.\n\n"
            "Capacity: This model has 4 toasting slots, each accommodating "
            "standard-width bread slices up to 1.2 inches thick, arranged "
            "in two independently controlled pairs.\n\n"
            "Temperature: The maximum toasting temperature is 230 degrees "
            "Celsius, with 6 selectable browning levels.\n\n"
            "Warranty: Model T3-2023 is covered by a 1-year limited "
            "warranty from the date of purchase, covering manufacturing "
            "defects only. Proof of purchase is required for claims.\n\n"
            "Safety: Always unplug the unit before cleaning. Do not "
            "immerse in water. Allow the unit to cool for at least 10 "
            "minutes before storage.\n\n"
            "Cleaning: The crumb tray slides out from the base and should "
            "be emptied after every 5 uses. Wipe the exterior with a damp "
            "cloth only.\n\n"
            "Included accessories: The box includes one bread rack "
            "attachment and a printed quick-start guide.\n\n"
            "Troubleshooting: If the toaster does not power on, check that "
            "it is fully seated in the outlet and the circuit breaker has "
            "not tripped. If slices toast unevenly, confirm both slot "
            "pairs are set to the same browning level."
        ),
        "queries": [
            "How many watts does the toaster use?",
            "How many toasting slots does it have?",
            "What is the maximum toasting temperature?",
            "How long is the warranty period?",
            "How often should the crumb tray be emptied?",
        ],
    },
    {
        "doc_id": "toaster_3000_pro_manual_t3p_2024.pdf",
        "text": (
            "Toaster 3000 Pro User Manual - Model T3P-2024\n\n"
            "Power: The Toaster 3000 Pro (Model T3P-2024) operates at 1200 "
            "watts on a standard 120V household circuit. Standby power "
            "draw is under 0.3 watts, meeting Energy Star idle requirements.\n\n"
            "Capacity: This model has 6 toasting slots, each accommodating "
            "standard or wide-width bread slices up to 1.5 inches thick, "
            "arranged in three independently controlled pairs.\n\n"
            "Temperature: The maximum toasting temperature is 250 degrees "
            "Celsius, with 8 selectable browning levels, including a "
            "dedicated bagel setting that toasts one side only.\n\n"
            "Warranty: Model T3P-2024 is covered by a 2-year limited "
            "warranty from the date of purchase, covering manufacturing "
            "defects and normal wear on heating elements.\n\n"
            "Safety: Always unplug the unit before cleaning. Do not "
            "immerse in water. Allow the unit to cool for at least 10 "
            "minutes before storage.\n\n"
            "Cleaning: The crumb tray slides out from the base and should "
            "be emptied after every 5 uses. Wipe the exterior with a damp "
            "cloth only.\n\n"
            "Included accessories: The box includes a bread rack "
            "attachment, a bagel rack attachment, and a printed "
            "quick-start guide.\n\n"
            "Troubleshooting: If the toaster does not power on, check that "
            "it is fully seated in the outlet and the circuit breaker has "
            "not tripped. If slices toast unevenly, confirm all three slot "
            "pairs are set to the same browning level."
        ),
        "queries": [
            "How many watts does the toaster use?",
            "How many toasting slots does it have?",
            "What is the maximum toasting temperature?",
            "How long is the warranty period?",
            "Does this model have a bagel setting?",
        ],
    },
    {
        "doc_id": "bella_vista_bistro_menu.pdf",
        "text": (
            "Bella Vista Bistro Dinner Menu\n\n"
            "Appetizers: Bruschetta with tomato and basil, $9. Calamari "
            "fritti with marinara, $12. Caprese salad with fresh "
            "mozzarella, $10. Soup of the day, $7.\n\n"
            "Main courses: Grilled salmon with lemon butter sauce, $24. "
            "Chicken parmesan with spaghetti, $19. Margherita pizza, "
            "wood-fired, $16. Beef tenderloin with red wine reduction, "
            "$32. Eggplant parmesan (vegetarian), $18.\n\n"
            "Pasta: Fettuccine alfredo, $17. Spaghetti carbonara, $18. "
            "Penne arrabbiata, $15. Lobster ravioli, $26.\n\n"
            "Desserts: Tiramisu, $8. Chocolate lava cake, $9. Cannoli, $7. "
            "Panna cotta with berry compote, $8.\n\n"
            "Beverages: House red or white wine, $9 per glass. Craft beer "
            "selection, $7. Espresso, $4. Italian soda, $5.\n\n"
            "Hours: Open Tuesday through Sunday, 5pm to 10pm. Closed "
            "Mondays. The bar stays open until 11pm on Friday and Saturday.\n\n"
            "Reservations: Recommended for parties of 6 or more. Walk-ins "
            "are welcome but may experience a wait during peak hours "
            "(7pm to 9pm on weekends).\n\n"
            "Private events: The back room seats up to 30 guests and can "
            "be booked for private events with 2 weeks notice.\n\n"
            "Contact: For reservations call (555) 234-5678 or book online "
            "at bellavistabistro.example."
        ),
        "queries": [
            "How much does the grilled salmon cost?",
            "What desserts are available?",
            "What are the restaurant's hours?",
            "Is the restaurant open on Mondays?",
            "How much is a glass of house wine?",
        ],
    },
    {
        "doc_id": "devtools_pro_license_agreement.pdf",
        "text": (
            "DevTools Pro Software License Agreement\n\n"
            "License grant: DevTools Pro grants the licensee a "
            "non-exclusive, non-transferable license to use the software "
            "on up to 3 devices per seat purchased.\n\n"
            "Subscription tiers: The Individual tier costs $15 per month "
            "and includes core debugging tools. The Team tier costs $45 "
            "per month per seat and adds collaborative code review. The "
            "Enterprise tier is custom-priced and includes SSO and audit "
            "logging.\n\n"
            "Restrictions: The licensee may not reverse engineer, "
            "decompile, or resell the software. Redistribution outside "
            "the licensee's organization is prohibited without written "
            "consent.\n\n"
            "Data and privacy: DevTools Pro collects anonymized usage "
            "telemetry by default, which can be disabled in Settings > "
            "Privacy. No source code is transmitted to DevTools Pro "
            "servers under any tier.\n\n"
            "Termination: Either party may terminate this agreement with "
            "30 days written notice. Upon termination, the licensee must "
            "uninstall all copies of the software within 14 days.\n\n"
            "Support: Individual tier includes community forum support. "
            "Team and Enterprise tiers include email support with a "
            "24-hour response time target.\n\n"
            "Updates: Minor version updates are included in all tiers. "
            "Major version upgrades are included for Team and Enterprise "
            "tiers; Individual tier users pay a discounted upgrade fee.\n\n"
            "Warranty disclaimer: The software is provided 'as is' "
            "without warranty of any kind, express or implied."
        ),
        "queries": [
            "How many devices can I use per seat?",
            "What is the monthly cost of the Team tier?",
            "Can I reverse engineer the software?",
            "How much notice is needed to terminate the agreement?",
            "What support does the Individual tier include?",
        ],
    },
    {
        "doc_id": "homeshield_insurance_2023.pdf",
        "text": (
            "HomeShield Insurance Policy - 2023 Term\n\n"
            "Coverage summary: This policy provides dwelling coverage up "
            "to $350,000, personal property coverage up to $175,000, and "
            "liability coverage up to $300,000 per occurrence.\n\n"
            "Annual premium: The total annual premium for this policy term "
            "is $1,240, payable monthly at $103.33 or annually with a 5 "
            "percent discount.\n\n"
            "Deductible: The standard deductible for wind and hail claims "
            "is $1,500. The deductible for all other covered perils is "
            "$1,000.\n\n"
            "Exclusions: This policy does not cover flood damage, "
            "earthquake damage, or damage from normal wear and tear. "
            "Flood coverage is available as a separate policy through the "
            "National Flood Insurance Program.\n\n"
            "Additional living expenses: If the home becomes uninhabitable "
            "due to a covered loss, this policy covers up to $35,000 in "
            "additional living expenses for up to 12 months.\n\n"
            "Claims process: Claims must be reported within 60 days of "
            "the loss. A claims adjuster will typically contact the "
            "policyholder within 3 business days.\n\n"
            "Discounts: Policyholders may qualify for a 10 percent "
            "discount for monitored security systems and a 5 percent "
            "discount for smoke detectors in every room.\n\n"
            "Renewal: This policy renews automatically each year unless "
            "cancelled in writing at least 30 days before the renewal date."
        ),
        "queries": [
            "What is the dwelling coverage limit?",
            "What is the annual premium?",
            "What is the deductible for wind and hail claims?",
            "How much does the additional living expenses coverage provide?",
            "How many days notice is needed to cancel before renewal?",
        ],
    },
    {
        "doc_id": "homeshield_insurance_2024.pdf",
        "text": (
            "HomeShield Insurance Policy - 2024 Term\n\n"
            "Coverage summary: This policy provides dwelling coverage up "
            "to $400,000, personal property coverage up to $200,000, and "
            "liability coverage up to $300,000 per occurrence.\n\n"
            "Annual premium: The total annual premium for this policy term "
            "is $1,460, payable monthly at $121.67 or annually with a 5 "
            "percent discount.\n\n"
            "Deductible: The standard deductible for wind and hail claims "
            "is $1,750. The deductible for all other covered perils is "
            "$1,000.\n\n"
            "Exclusions: This policy does not cover flood damage, "
            "earthquake damage, or damage from normal wear and tear. "
            "Flood coverage is available as a separate policy through the "
            "National Flood Insurance Program.\n\n"
            "Additional living expenses: If the home becomes uninhabitable "
            "due to a covered loss, this policy covers up to $40,000 in "
            "additional living expenses for up to 12 months.\n\n"
            "Claims process: Claims must be reported within 60 days of "
            "the loss. A claims adjuster will typically contact the "
            "policyholder within 3 business days.\n\n"
            "Discounts: Policyholders may qualify for a 10 percent "
            "discount for monitored security systems and a 5 percent "
            "discount for smoke detectors in every room.\n\n"
            "Renewal: This policy renews automatically each year unless "
            "cancelled in writing at least 30 days before the renewal date."
        ),
        "queries": [
            "What is the dwelling coverage limit?",
            "What is the annual premium?",
            "What is the deductible for wind and hail claims?",
            "How much does the additional living expenses coverage provide?",
            "How many days notice is needed to cancel before renewal?",
        ],
    },
    {
        "doc_id": "cascade_bank_savings_account_terms.pdf",
        "text": (
            "Cascade Bank Savings Account Terms and Conditions\n\n"
            "Interest rate: The Cascade Bank Standard Savings account "
            "earns an annual percentage yield of 2.15 percent, compounded "
            "daily and credited monthly.\n\n"
            "Minimum balance: A minimum daily balance of $300 is required "
            "to avoid a $5 monthly maintenance fee. Balances below $100 "
            "for more than 60 consecutive days may result in account "
            "closure.\n\n"
            "Withdrawal limits: Federal regulations and Cascade Bank "
            "policy limit certain types of withdrawals and transfers to 6 "
            "per statement cycle. Exceeding this limit twice in a rolling "
            "12-month period may result in conversion to a checking "
            "account.\n\n"
            "Opening deposit: A minimum opening deposit of $25 is required "
            "to open a Standard Savings account.\n\n"
            "ATM access: Savings account holders receive a debit card "
            "usable at any Cascade Bank ATM free of charge. Out-of-network "
            "ATM withdrawals incur a $2.50 fee per transaction.\n\n"
            "Overdraft protection: Savings accounts can be linked to a "
            "Cascade Bank checking account for automatic overdraft "
            "protection transfers, in increments of $50.\n\n"
            "Account closure: Accounts closed within 90 days of opening "
            "are subject to a $25 early closure fee.\n\n"
            "Contact: Customer service is available at 1-800-555-0142, "
            "Monday through Saturday, 8am to 8pm."
        ),
        "queries": [
            "What is the annual percentage yield on the savings account?",
            "What is the minimum balance required to avoid a monthly fee?",
            "How many withdrawals are allowed per statement cycle?",
            "What is the minimum opening deposit?",
            "What is the early closure fee?",
        ],
    },
    {
        "doc_id": "cascade_bank_checking_account_terms.pdf",
        "text": (
            "Cascade Bank Checking Account Terms and Conditions\n\n"
            "Interest rate: The Cascade Bank Free Checking account does "
            "not earn interest. The Cascade Bank Interest Checking account "
            "earns an annual percentage yield of 0.25 percent.\n\n"
            "Minimum balance: Free Checking has no minimum balance "
            "requirement. Interest Checking requires a minimum daily "
            "balance of $1,500 to avoid a $12 monthly maintenance fee.\n\n"
            "Overdraft: Standard overdraft coverage charges a $34 fee per "
            "item, up to 3 fees per day. Overdraft protection transfers "
            "from a linked savings account are free.\n\n"
            "Opening deposit: A minimum opening deposit of $50 is required "
            "for either checking account type.\n\n"
            "Debit card: All checking accounts include a debit card with "
            "no annual fee. Daily purchase limits are $2,500 and daily ATM "
            "withdrawal limits are $500 by default, adjustable on request.\n\n"
            "Direct deposit: Setting up direct deposit of at least $500 "
            "per month waives the monthly maintenance fee on Interest "
            "Checking automatically.\n\n"
            "Mobile check deposit: Available for accounts open more than "
            "30 days, with a daily deposit limit of $5,000.\n\n"
            "Contact: Customer service is available at 1-800-555-0142, "
            "Monday through Saturday, 8am to 8pm."
        ),
        "queries": [
            "Does the Free Checking account earn interest?",
            "What is the overdraft fee per item?",
            "What is the minimum opening deposit for a checking account?",
            "What is the daily ATM withdrawal limit?",
            "How much direct deposit is needed to waive the monthly fee?",
        ],
    },
    {
        "doc_id": "peak_fitness_membership_agreement.pdf",
        "text": (
            "Peak Fitness Club Membership Agreement\n\n"
            "Membership tiers: The Basic tier costs $29 per month and "
            "includes access to one home club location. The All-Access "
            "tier costs $49 per month and includes access to all Peak "
            "Fitness locations nationwide. The Premium tier costs $79 per "
            "month and adds unlimited group classes and 2 guest passes "
            "per month.\n\n"
            "Enrollment fee: A one-time enrollment fee of $49 applies to "
            "all new memberships, waived during promotional periods.\n\n"
            "Contract terms: Month-to-month memberships may be cancelled "
            "with 30 days written notice. Annual memberships receive a 15 "
            "percent discount but require a 12-month commitment with an "
            "early termination fee of $150.\n\n"
            "Freeze policy: Members may freeze their membership for "
            "medical reasons or travel for up to 3 months per year at no "
            "cost, with documentation required for medical freezes.\n\n"
            "Guest policy: Basic and All-Access members may bring a guest "
            "for a $15 day-pass fee. Premium members receive 2 free guest "
            "passes per month.\n\n"
            "Hours: Clubs are open 5am to 11pm on weekdays and 7am to 9pm "
            "on weekends. Some 24-hour locations are available for "
            "All-Access and Premium members.\n\n"
            "Personal training: Personal training sessions are billed "
            "separately at $65 per session, with package discounts "
            "available for 10 or more sessions.\n\n"
            "Cancellation: Cancellation requests must be submitted in "
            "writing at the front desk or through the member portal."
        ),
        "queries": [
            "How much does the All-Access tier cost per month?",
            "What is the one-time enrollment fee?",
            "What is the early termination fee for annual memberships?",
            "How long can a membership be frozen per year?",
            "How much does a personal training session cost?",
        ],
    },
    {
        "doc_id": "urban_property_lease_agreement.pdf",
        "text": (
            "Urban Property Group Residential Lease Agreement\n\n"
            "Lease term: This lease is for a 12-month term beginning on "
            "the move-in date specified in the lease summary, with "
            "automatic conversion to month-to-month at a 10 percent rent "
            "premium if not renewed.\n\n"
            "Rent: Monthly rent is due on the 1st of each month. Rent paid "
            "after the 5th incurs a late fee of $75 plus $10 per "
            "additional day.\n\n"
            "Security deposit: A security deposit equal to one month's "
            "rent is required at signing, refundable within 21 days of "
            "move-out minus any deductions for damage beyond normal wear.\n\n"
            "Pet policy: Pets are allowed with a non-refundable pet fee of "
            "$300 per pet and an additional $35 monthly pet rent. Maximum "
            "of 2 pets per unit, with breed restrictions for dogs over 50 "
            "pounds.\n\n"
            "Utilities: Tenant is responsible for electricity, gas, and "
            "internet. Water, sewer, and trash are included in rent up to "
            "a monthly cap of $60, with overages billed to the tenant.\n\n"
            "Maintenance: Non-emergency maintenance requests are addressed "
            "within 3 business days. Emergency requests (no heat, no "
            "water, or active leaks) are addressed within 24 hours.\n\n"
            "Subletting: Subletting requires written approval from "
            "management at least 30 days in advance and is subject to a "
            "$100 processing fee.\n\n"
            "Parking: One assigned parking spot is included; additional "
            "spots are available for $50 per month, subject to "
            "availability."
        ),
        "queries": [
            "What is the late fee for rent paid after the 5th?",
            "How much is the security deposit?",
            "What is the pet fee and monthly pet rent?",
            "How quickly are emergency maintenance requests addressed?",
            "How much does an additional parking spot cost?",
        ],
    },
    {
        "doc_id": "skyline_airlines_baggage_policy.pdf",
        "text": (
            "Skyline Airlines Baggage Policy\n\n"
            "Carry-on allowance: Passengers may bring one carry-on bag up "
            "to 22 x 14 x 9 inches and one personal item free of charge on "
            "all fare types.\n\n"
            "Checked baggage: Basic Economy fares do not include a free "
            "checked bag. Standard Economy and above include one free "
            "checked bag up to 50 pounds. A second checked bag costs $40 "
            "domestically or $75 internationally.\n\n"
            "Overweight fees: Bags between 51 and 70 pounds incur a $100 "
            "overweight fee. Bags over 70 pounds are not accepted for "
            "standard check-in and require special cargo arrangements.\n\n"
            "Oversized items: Items exceeding 62 linear inches (length "
            "plus width plus height) incur a $150 oversized fee, except "
            "for approved sporting equipment such as golf bags and ski "
            "equipment, which have a flat $50 fee.\n\n"
            "Fragile items: Musical instruments and fragile items may be "
            "carried on if they fit within carry-on dimensions, or checked "
            "with a signed liability waiver.\n\n"
            "Delayed baggage: Skyline Airlines provides a $50 per day "
            "reimbursement for essential items if checked baggage is "
            "delayed more than 12 hours, up to a maximum of $300.\n\n"
            "Lost baggage claims: Claims for lost baggage must be filed "
            "within 21 days of travel. Compensation is capped at $3,800 "
            "for domestic flights per the standard liability limit.\n\n"
            "Special items: Wheelchairs and mobility devices are carried "
            "free of charge in addition to standard baggage allowance."
        ),
        "queries": [
            "How many free checked bags does Standard Economy include?",
            "What is the fee for a second checked bag domestically?",
            "What is the overweight fee for bags between 51 and 70 pounds?",
            "How much daily reimbursement is provided for delayed baggage?",
            "What is the compensation cap for lost baggage on domestic flights?",
        ],
    },
    {
        "doc_id": "skyline_airlines_loyalty_program.pdf",
        "text": (
            "Skyline Airlines SkyMiles Loyalty Program Terms\n\n"
            "Earning miles: Members earn 5 miles per dollar spent on "
            "Basic Economy fares, 8 miles per dollar on Standard Economy, "
            "and 12 miles per dollar on Business fares.\n\n"
            "Elite tiers: Silver status requires 25,000 miles or 30 "
            "flights per year. Gold status requires 50,000 miles or 60 "
            "flights per year. Platinum status requires 100,000 miles or "
            "100 flights per year.\n\n"
            "Silver benefits: Priority boarding and one free checked bag "
            "regardless of fare type.\n\n"
            "Gold benefits: All Silver benefits plus complimentary seat "
            "upgrades when available and access to Skyline Lounges.\n\n"
            "Platinum benefits: All Gold benefits plus guaranteed seat "
            "upgrades on domestic flights and a dedicated concierge phone "
            "line.\n\n"
            "Mile expiration: Miles expire after 24 months of account "
            "inactivity. Any earning or redemption activity resets the "
            "expiration clock.\n\n"
            "Redemption: A domestic round-trip award ticket starts at "
            "25,000 miles in Basic Economy or 60,000 miles in Business "
            "class, subject to availability.\n\n"
            "Companion pass: Platinum members receive one companion pass "
            "per year valid for a free companion ticket on any domestic "
            "flight, taxes and fees excluded.\n\n"
            "Family pooling: Up to 5 family members may pool miles into a "
            "single household account at no cost."
        ),
        "queries": [
            "How many miles are needed for Gold status?",
            "What benefits does Silver status include?",
            "How many miles does a domestic round-trip award ticket start at?",
            "When do miles expire?",
            "What does the Platinum companion pass include?",
        ],
    },
    {
        "doc_id": "greenfield_university_admissions_policy.pdf",
        "text": (
            "Greenfield University Undergraduate Admissions Policy\n\n"
            "Application deadlines: Early Decision applications are due "
            "November 1 with decisions released by December 15. Regular "
            "Decision applications are due January 15 with decisions "
            "released by March 31.\n\n"
            "Required materials: Applicants must submit transcripts, two "
            "letters of recommendation, a personal essay, and either SAT "
            "or ACT scores, though test-optional review is available for "
            "the 2024-2025 cycle.\n\n"
            "GPA and coursework: The middle 50 percent of admitted "
            "students have a weighted GPA between 3.7 and 4.2. Applicants "
            "are expected to have completed at least 4 years of English "
            "and 3 years each of math, science, and social studies.\n\n"
            "Application fee: A $65 application fee is required, with fee "
            "waivers available for demonstrated financial need.\n\n"
            "Transfer admissions: Transfer applicants must have completed "
            "at least 24 college credit hours with a minimum GPA of 3.0 to "
            "be considered.\n\n"
            "International students: International applicants must submit "
            "TOEFL scores of at least 90 or IELTS scores of at least 6.5, "
            "along with proof of financial support for the full cost of "
            "attendance.\n\n"
            "Waitlist: Waitlisted students are notified by April 1 and may "
            "be admitted on a rolling basis through August 1 as space "
            "becomes available.\n\n"
            "Deferral: Admitted students may request a one-year deferral "
            "for documented gap year plans, subject to admissions "
            "committee approval."
        ),
        "queries": [
            "When is the Regular Decision application deadline?",
            "What is the application fee?",
            "What GPA range do the middle 50 percent of admitted students have?",
            "What TOEFL score do international applicants need?",
            "How many college credit hours are required for transfer admission?",
        ],
    },
    {
        "doc_id": "greenfield_university_financial_aid_policy.pdf",
        "text": (
            "Greenfield University Financial Aid Policy\n\n"
            "FAFSA priority deadline: Students must submit the FAFSA by "
            "February 15 to be considered for the full range of "
            "need-based aid, including institutional grants.\n\n"
            "Need-based grants: Greenfield University meets 100 percent of "
            "demonstrated financial need for admitted undergraduate "
            "students through a combination of grants, work-study, and "
            "federal loans.\n\n"
            "Merit scholarships: The Presidential Scholarship awards "
            "$25,000 per year to students with a weighted GPA above 4.0 "
            "and a strong record of extracurricular leadership. The Dean's "
            "Scholarship awards $15,000 per year for a weighted GPA above "
            "3.8.\n\n"
            "Work-study: Federal Work-Study awards typically range from "
            "$2,000 to $3,500 per year, paid biweekly for hours worked in "
            "approved campus positions.\n\n"
            "Loan options: The university packages the Federal Direct "
            "Subsidized and Unsubsidized Loans as part of standard aid "
            "offers, with a maximum of $5,500 in federal loans for "
            "first-year students.\n\n"
            "Satisfactory academic progress: Students must maintain a "
            "cumulative GPA of at least 2.0 and complete at least 67 "
            "percent of attempted credit hours to remain eligible for aid.\n\n"
            "Appeals: Students may appeal a financial aid decision within "
            "30 days of notification by submitting a written appeal with "
            "supporting documentation to the Office of Financial Aid.\n\n"
            "Outside scholarships: Students must report outside "
            "scholarships, which may result in a reduction of "
            "need-based grant aid on a dollar-for-dollar basis above "
            "$3,000 combined."
        ),
        "queries": [
            "What is the FAFSA priority deadline?",
            "How much does the Presidential Scholarship award per year?",
            "What is the typical range for Federal Work-Study awards?",
            "What GPA is required to maintain satisfactory academic progress?",
            "How many days do students have to appeal a financial aid decision?",
        ],
    },
    {
        "doc_id": "northstar_logistics_shipping_terms.pdf",
        "text": (
            "Northstar Logistics Shipping Terms of Service\n\n"
            "Standard shipping: Standard ground shipping takes 5 to 7 "
            "business days and costs a flat $6.99 for packages under 5 "
            "pounds, with weight-based pricing above that.\n\n"
            "Expedited shipping: 2-day shipping costs $14.99 and overnight "
            "shipping costs $24.99, both available for orders placed "
            "before 2pm local time.\n\n"
            "International shipping: International orders take 10 to 21 "
            "business days depending on destination and are subject to "
            "customs duties paid by the recipient.\n\n"
            "Free shipping threshold: Orders over $75 qualify for free "
            "standard ground shipping within the continental United "
            "States.\n\n"
            "Package tracking: All shipments include tracking numbers "
            "emailed within 24 hours of dispatch, with updates every 12 "
            "hours until delivery.\n\n"
            "Signature requirements: Orders over $500 require a signature "
            "upon delivery. Orders under $500 may be left at the door per "
            "carrier discretion.\n\n"
            "Address changes: Address changes are only possible within 1 "
            "hour of order placement, after which the order enters "
            "fulfillment and cannot be redirected.\n\n"
            "Weather delays: Northstar Logistics is not liable for delays "
            "caused by severe weather but will provide updated delivery "
            "estimates through the tracking portal.\n\n"
            "Damaged shipments: Damage must be reported within 48 hours of "
            "delivery with photo documentation to qualify for replacement."
        ),
        "queries": [
            "How long does standard ground shipping take?",
            "How much does overnight shipping cost?",
            "What is the free shipping threshold?",
            "Do orders over $500 require a signature?",
            "How soon must damaged shipments be reported?",
        ],
    },
    {
        "doc_id": "northstar_logistics_returns_policy.pdf",
        "text": (
            "Northstar Logistics Returns and Refunds Policy\n\n"
            "Return window: Most items may be returned within 30 days of "
            "delivery for a full refund. Electronics have a shorter 15-day "
            "return window.\n\n"
            "Condition requirements: Returned items must be unused, in "
            "original packaging, with all tags and accessories included. "
            "Items showing signs of use may only qualify for partial "
            "refunds.\n\n"
            "Return shipping: Return shipping is free for defective or "
            "incorrect items. For other returns, a $5.99 return shipping "
            "fee is deducted from the refund unless the customer used a "
            "prepaid Northstar return label.\n\n"
            "Refund processing: Refunds are issued to the original payment "
            "method within 5 to 7 business days of the returned item being "
            "received at the warehouse.\n\n"
            "Exchanges: Size and color exchanges are processed as a free "
            "swap without additional shipping charges, subject to stock "
            "availability.\n\n"
            "Final sale items: Items marked 'final sale' and gift cards "
            "are not eligible for return or refund.\n\n"
            "Restocking fee: Large furniture items are subject to a 15 "
            "percent restocking fee on returns that are not due to defect "
            "or damage.\n\n"
            "Gift returns: Items received as gifts can be returned for "
            "store credit without needing the original receipt, using the "
            "gift receipt or order confirmation email.\n\n"
            "Late returns: Returns initiated after the window closes may "
            "be accepted at Northstar's discretion for store credit only, "
            "minus a 20 percent restocking fee."
        ),
        "queries": [
            "What is the standard return window?",
            "What is the return window for electronics?",
            "How much is the return shipping fee for non-defective items?",
            "How long does refund processing take?",
            "What is the restocking fee on large furniture returns?",
        ],
    },
]


@dataclass
class QueryResult:
    doc_id: str
    scenario: str  # "isolated" or "shared"
    n_docs_in_index: int
    query: str
    contamination_ratio: float
    confidence_mismatch_keyword: bool
    confidence_mismatch_llm: bool
    rri_keyword: float
    rri_llm: float
    tier_keyword: int
    tier_llm: int
    answer: str
    retrieved_sources: list = field(default_factory=list)


def run_query(vs, query, current_doc_id, scenario, n_docs_in_index) -> QueryResult:
    retrieved = retrieve(vs, query, k=TOP_K)
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
        doc_id=current_doc_id,
        scenario=scenario,
        n_docs_in_index=n_docs_in_index,
        query=query,
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


def summarize(results: list[QueryResult], label: str) -> dict:
    n = len(results)
    tier_counts = {1: 0, 2: 0, 3: 0}
    for r in results:
        tier_counts[r.tier_keyword] += 1
    intercepted = tier_counts[2] + tier_counts[3]
    avg_contam = statistics.mean(r.contamination_ratio for r in results) if results else 0.0
    print(f"  [{label}] n={n} avg_contam={avg_contam:.3f} tiers={tier_counts} "
          f"interception={intercepted}/{n}={intercepted/n:.0%}" if n else f"  [{label}] n=0")
    return {"n": n, "avg_contamination": avg_contam, "tier_counts": tier_counts,
            "interception_rate": intercepted / n if n else 0.0}


def main():
    all_results: list[QueryResult] = []

    print(f"=== Ingesting {len(DOCUMENTS)} documents ===")
    all_chunks = {}
    for doc in DOCUMENTS:
        chunks = ingest_pdf(make_pdf_bytes(doc["doc_id"], doc["text"]), doc["doc_id"])
        all_chunks[doc["doc_id"]] = chunks
        print(f"  {doc['doc_id']} -> {len(chunks)} chunks")

    # --- Isolated baseline: scale-invariant, run once per document ---
    print("\n=== Isolated baseline (one document per index) ===")
    for doc in DOCUMENTS:
        vs_isolated = FAISS.from_documents(all_chunks[doc["doc_id"]], embeddings)
        for q in doc["queries"]:
            r = run_query(vs_isolated, q, doc["doc_id"], "isolated", n_docs_in_index=1)
            all_results.append(r)
    print(f"  {len(DOCUMENTS) * 5} isolated queries complete")

    # --- Growth experiment: shared index built incrementally, 5 -> 10 -> 20 ---
    print("\n=== Growth experiment: shared, never-cleared index ===")
    probe_docs = DOCUMENTS[:5]
    probe_queries = [(d["doc_id"], q) for d in probe_docs for q in d["queries"]]

    vs_shared = None
    cumulative_chunks = 0
    checkpoints = [(5, DOCUMENTS[:5]), (10, DOCUMENTS[5:10]), (20, DOCUMENTS[10:20])]

    corpus_stats = []
    for n_docs, new_docs in checkpoints:
        for doc in new_docs:
            chunks = all_chunks[doc["doc_id"]]
            cumulative_chunks += len(chunks)
            if vs_shared is None:
                vs_shared = FAISS.from_documents(chunks, embeddings)
            else:
                vs_shared.add_documents(chunks)

        ratio = FETCH_K / cumulative_chunks
        corpus_stats.append({"n_docs": n_docs, "total_chunks": cumulative_chunks, "fetch_k": FETCH_K, "ratio": ratio})
        print(f"\n--- Checkpoint: {n_docs} documents, {cumulative_chunks} total chunks, "
              f"fetch_k={FETCH_K}, fetch_k/corpus={ratio:.2f} ---")

        # Fixed probe set (docs 1-5's 25 queries), run at every checkpoint.
        probe_results = []
        for doc_id, q in probe_queries:
            r = run_query(vs_shared, q, doc_id, "shared", n_docs_in_index=n_docs)
            r_marked = r
            probe_results.append(r_marked)
        all_results.extend(probe_results)
        summarize(probe_results, f"probe set @ {n_docs} docs")

        # Full coverage: the newly-added documents' own queries, run once
        # (at the checkpoint where they first exist), except at n_docs=5
        # where new_docs == probe_docs and we'd just be duplicating the
        # probe set.
        if n_docs != 5:
            new_doc_results = []
            for doc in new_docs:
                for q in doc["queries"]:
                    r = run_query(vs_shared, q, doc["doc_id"], "shared", n_docs_in_index=n_docs)
                    new_doc_results.append(r)
            all_results.extend(new_doc_results)
            summarize(new_doc_results, f"new docs' own queries @ {n_docs} docs")

    # --- Report ---
    print("\n\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    isolated = [r for r in all_results if r.scenario == "isolated"]
    print(f"\n[ISOLATED BASELINE] n={len(isolated)}")
    fp = [r for r in isolated if r.tier_keyword != 1]
    print(f"  false positives (should be 0): {len(fp)}")
    for r in fp:
        print(f"    '{r.query[:45]}' ({r.doc_id}) contam={r.contamination_ratio} tier={r.tier_keyword}")

    print("\n[GROWTH EXPERIMENT: fixed 25-query probe set, same queries at every scale]")
    print(f"{'n_docs':>8} {'chunks':>8} {'fetch_k/corpus':>15} {'avg_contam':>11} {'interception':>13}")
    for stat in corpus_stats:
        probe_subset = [r for r in all_results if r.scenario == "shared" and r.n_docs_in_index == stat["n_docs"]
                         and (r.doc_id, r.query) in probe_queries]
        s = summarize(probe_subset, "recap") if False else None
        n = len(probe_subset)
        avg_c = statistics.mean(r.contamination_ratio for r in probe_subset) if probe_subset else 0
        tiers = Counter(r.tier_keyword for r in probe_subset)
        intercepted = sum(v for k, v in tiers.items() if k > 1)
        print(f"{stat['n_docs']:>8} {stat['total_chunks']:>8} {stat['ratio']:>15.2f} {avg_c:>11.3f} "
              f"{f'{intercepted}/{n}={intercepted/n:.0%}' if n else 'n/a':>13}")

    print("\n[FULL COVERAGE: every document's own queries, at the scale where it exists]")
    full_shared = [r for r in all_results if r.scenario == "shared"]
    # Dedupe: keep each (doc_id, query) result only from the checkpoint where
    # that document was newly added (its "native" scale), to avoid
    # double-counting the probe set across checkpoints in this view.
    native_scale = {}
    for d in DOCUMENTS[:5]:
        native_scale[d["doc_id"]] = 5
    for d in DOCUMENTS[5:10]:
        native_scale[d["doc_id"]] = 10
    for d in DOCUMENTS[10:20]:
        native_scale[d["doc_id"]] = 20
    full_coverage = [r for r in full_shared if r.n_docs_in_index == native_scale[r.doc_id]]
    print(f"{'n_docs':>8} {'chunks':>8} {'fetch_k/corpus':>15} {'avg_contam':>11} {'interception':>13}")
    for stat in corpus_stats:
        subset = [r for r in full_coverage if r.n_docs_in_index == stat["n_docs"]]
        n = len(subset)
        avg_c = statistics.mean(r.contamination_ratio for r in subset) if subset else 0
        tiers = Counter(r.tier_keyword for r in subset)
        intercepted = sum(v for k, v in tiers.items() if k > 1)
        print(f"{stat['n_docs']:>8} {stat['total_chunks']:>8} {stat['ratio']:>15.2f} {avg_c:>11.3f} "
              f"{f'{intercepted}/{n}={intercepted/n:.0%}' if n else 'n/a':>13}")

    contam_all = [r for r in all_results if r.scenario == "shared"]
    agree = sum(1 for r in contam_all if r.tier_keyword == r.tier_llm)
    print(f"\nKeyword vs LLM-judge tier agreement (all shared-index cases): {agree}/{len(contam_all)}")

    boundary_cases = [r for r in contam_all if abs(r.contamination_ratio - 0.25) < 1e-9]
    print(f"\nExactly-25%-contamination cases observed naturally: {len(boundary_cases)}")
    for r in boundary_cases:
        print(f"  '{r.query[:45]}' ({r.doc_id}, n_docs={r.n_docs_in_index}) tier={r.tier_keyword} "
              f"rri={r.rri_keyword} confidence_mismatch={r.confidence_mismatch_keyword}")

    print("\nContamination ratio distribution (shared-index, all queries):")
    dist = Counter(r.contamination_ratio for r in contam_all)
    for ratio in sorted(dist):
        print(f"  {ratio:.2f}: {dist[ratio]} queries")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in all_results], f, indent=2)
    print(f"\nRaw results ({len(all_results)} total) written to eval_results.json")


if __name__ == "__main__":
    main()
