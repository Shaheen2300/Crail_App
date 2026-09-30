# CRAIL: Contextual Risk-Aware Intervention Layer

A RAG app (PDF upload → FAISS → OpenAI generation) with a human-in-the-loop
safety layer that catches **Phantom Context Hallucination**: stale chunks
from a previously uploaded document silently contaminating an answer about
the current one, because the vector index wasn't cleared between uploads.

CRAIL sits after retrieval and before the answer reaches the user. It
computes a Retrieval Risk Index (RRI) from two signals: how much of the
retrieved context is actually from the current document (provenance), and
whether the answer sounds confidently source-grounded (confidence). It then
routes the response: deliver it, hold it for a human reviewer, or block it
outright.

## Research context

CRAIL is the working prototype behind *Silent Poisoning: Stale Context
Persistence, Adversarial Injection, and Human Trust Miscalibration in
Retrieval-Augmented Generation Systems* (Mishra & Memon, 2026; working
paper in preparation for journal submission).

The paper's 60-query pilot covered 6 document pairs × 10 queries across
policy, healthcare, and technical domains, using an earlier prototype. In
that pilot, **73.3%** of queries (44/60) retrieved chunks from the wrong
document with no visible warning to the user, and CRAIL intercepted
**72.7%** (32/44) of those contaminated outputs. The misses from that pilot
led to the threshold fix described under *Key implementation choices*
below. This repo is the rebuilt app plus the larger evaluation harnesses.

## How it works

```
PDF upload ─► chunk + stamp source_doc ─► FAISS ─► top-k retrieval (MMR)
                                                   │
                     ┌─────────────────────────────┘
                     ▼
     provenance: contamination ratio ─┐
     confidence: keyword or LLM judge ┴─► RRI score ─► Tier 1 deliver
                                                      Tier 2 hold for human review
                                                      Tier 3 block, human reply required
                                                            │
                                                            ▼
                                                  review queue + audit log
```

**Stack:** Python 3.11, Streamlit, LangChain, FAISS, OpenAI
(`gpt-4o-mini` chat and judge, `text-embedding-3-small` embeddings; each
can be overridden in `.env`), Plotly.

| Path | What it does |
|---|---|
| `app.py` | Streamlit UI: Ask, Review Queue, and Audit Log tabs |
| `config.py` | Every tunable constant; reads secrets from `.env` or Streamlit secrets |
| `crail/ingestion.py`, `vectorstore.py`, `retrieval.py`, `generation.py` | RAG pipeline |
| `crail/provenance.py`, `confidence.py`, `rri.py` | The three CRAIL signals and the risk score |
| `crail/review_queue.py`, `audit.py` | Human-in-the-loop queue and decision log |
| `evaluate*.py` | Offline evaluation harnesses (in-memory index; they never touch the app's live data) |

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then edit .env and add your OPENAI_API_KEY
# macOS/Linux: python3.11 -m venv .venv && source .venv/bin/activate && cp .env.example .env
streamlit run app.py
```

## Trying it out

1. **Clean case (Tier 1):** upload one PDF, ask a question about it. Index
   contamination is 0, answer is delivered directly.
2. **Reproduce the vulnerability (Tier 2/3):** upload document A. Then
   upload document B with **"Clear existing index before adding" unchecked**
   in the sidebar. This simulates the exact bug CRAIL exists to catch.
   Select document B as your "current document" and ask a question. If any
   of the top-4 retrieved chunks come from document A, you'll see a nonzero
   contamination ratio and, if the answer sounds confidently grounded, a
   Tier 2 or Tier 3 routing instead of a silently wrong answer.
3. **Review Queue tab:** Tier 2 items can be approved as-is, approved with
   an added transparency note, or escalated to Tier 3. Tier 3 items require
   a human-composed reply; nothing automatic is ever delivered for those.
4. **Audit Log tab:** every decision (including silent Tier 1 ones) is
   logged, with a running interception rate and tier breakdown.

## Key implementation choices

- `config.py` holds every tunable constant (chunk size/overlap, top_k,
  RRI weights, tier thresholds) in one place.
- `crail/provenance.py` measures contamination from a hard `source_doc`
  metadata label stamped on each chunk at ingestion time
  (`crail/ingestion.py`), never re-derived from chunk content. That's what
  makes it an objective measurement rather than a judgment call.
- `crail/confidence.py` offers two interchangeable detectors: a broadened
  keyword list, and an LLM-judge call (toggle in the sidebar, LLM judge is
  the default). They only affect whether an already-flagged response lands
  in Tier 2 or Tier 3, never whether it gets intercepted at all; that's
  driven entirely by the contamination ratio. **Use the LLM judge unless
  you have a specific reason not to**: across 200 real evaluated queries,
  the keyword method agreed with the LLM judge on severity 0 times. Modern
  models answer factually without ever using a listed trigger phrase.
- `crail/rri.py`: with the default weights (contamination 0.6, confidence
  0.4) and thresholds (Tier 2 ≥ 0.15, Tier 3 ≥ 0.7), **a response can never
  reach Tier 3 on contamination alone**. Max score without a confidence
  mismatch is `1.0 × 0.6 = 0.6`. Reaching Tier 3 always requires both high
  contamination and a detected confidence mismatch. If you change the
  weights/thresholds, re-derive this ceiling for your own values.
- With `TOP_K = 4`, one contaminated chunk out of four is always exactly
  25% contamination, giving `RRI = 0.25 × 0.6 = 0.15` on contamination
  alone. `TIER2_THRESHOLD` is set at exactly that value so **any measurable
  contamination reaches at least Tier 2, independent of confidence
  detection working at all.** This was not always true: the threshold used
  to be 0.3, and the confidence-bonus floor used to require contamination
  strictly greater than 0.25. Both constants sat exactly on top of the
  smallest real value TOP_K=4 can produce, so a single stray chunk was a
  mathematically guaranteed silent miss, confirmed empirically: a
  low-overlap document pair produced exactly this case on every query in
  testing and was delivered as Tier 1 every single time, regardless of how
  the answer was worded. If you change `TOP_K`, recompute
  `TIER2_THRESHOLD` as at most `CONTAMINATION_WEIGHT × (1 / TOP_K)`.

## Limitations (carried over from the design doc, updated after evaluation)

- The 0.6/0.4 weights are heuristic starting points, not empirically
  calibrated. The thresholds are no longer arbitrary: `TIER2_THRESHOLD` is
  derived from `TOP_K` as described above.
- Confidence detection only ever affects the hold-vs-block decision, never
  whether something is intercepted in the first place, and after the fix
  above that's true in the strongest sense: at TOP_K=4, contamination alone
  is now sufficient to guarantee interception at Tier 2, so confidence
  detection reliability no longer matters for catching contamination at
  all, only for how severely it's flagged.
- Interception performance still varies by document-pair type and by how
  many chunks each document contributes relative to `TOP_K`; don't treat
  any single run's interception rate as a general number. See `evaluate.py`
  for the harness used to measure this.

## Evaluation harnesses

Each script uses the real pipeline: real PDFs, embeddings, generation, and
RRI scoring. Results go to git-ignored `eval_results*.json` files.

| Script | Question it answers |
|---|---|
| `evaluate_pairs.py` | Regression check for the Tier 2 blind spot on the original 3 document pairs |
| `evaluate.py` | Does contamination stay stable as a shared index grows from 5 to 10 to 20 documents? |
| `evaluate_borderline.py` | Can a zero-tolerance threshold tell harmful contamination from benign cross-document overlap? |
| `evaluate_real_docs.py` | The same test on large real documents: is near-universal flagging a small-corpus artifact? |
| `evaluate_broad_retrieval.py` | Summary/analysis queries: clean on isolated indexes, and correctly flagged on a shared, never-cleared index? |

Also runs in GitHub Codespaces: the `.devcontainer` installs the
requirements and starts Streamlit automatically.
