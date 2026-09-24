"""CRAIL demo app: RAG pipeline with the Contextual Risk-Aware Intervention
Layer sitting between retrieval and the user.

Tabs:
  - Ask: upload docs, ask questions, see Tier 1 responses delivered live.
  - Review Queue: human-in-the-loop review for Tier 2 (hold) and Tier 3
    (block) responses.
  - Audit Log: every decision CRAIL has made, including the silent Tier 1
    ones, plus basic interception metrics.
"""
import html as html_lib
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

from config import (
    BROAD_TOP_K,
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONFIDENCE_MISMATCH_CONTAMINATION_FLOOR,
    CONFIDENCE_WEIGHT,
    CONTAMINATION_WEIGHT,
    EMBEDDING_MODEL,
    JUDGE_MODEL,
    OPENAI_API_KEY,
    TIER2_THRESHOLD,
    TIER3_THRESHOLD,
    TOP_K,
)
from crail import audit, registry, review_queue, vectorstore
from crail.attribution import attribute_sentences
from crail.generation import generate_answer, generate_answer_from_texts
from crail.grounding import check_grounding
from crail.ingestion import ingest_pdf
from crail.intent import classify_intent
from crail.retrieval import FILTER_FETCH_K, retrieve, retrieve_broad, retrieve_scoped
from crail.rri import crail_decision
from crail.sensitivity import sweep_contamination_weight, sweep_tier2_threshold
from crail.viz import get_all_chunks_with_vectors, project_2d

st.set_page_config(page_title="CRAIL", page_icon=":material/shield:", layout="wide")

# Validated colorblind-safe categorical palette (dataviz skill reference).
# Assigned in fixed order per doc_id, never cycled/reassigned.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
STATUS_COLORS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def get_client() -> OpenAI:
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return st.session_state.openai_client


def render_match_badge(source: str, current_doc_id: str) -> None:
    if source == current_doc_id:
        st.badge("matches current document", icon=":material/check_circle:", color="green")
    else:
        st.badge("mismatch: different document", icon=":material/error:", color="red")


def render_chunk(doc, current_doc_id: str) -> None:
    source = doc.metadata.get("source_doc", "UNKNOWN")
    with st.container(border=True):
        top = st.container(horizontal=True, vertical_alignment="center")
        top.caption(f"source_doc: `{source}`")
        with top:
            render_match_badge(source, current_doc_id)
        st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))


def render_chunk_dict(chunk: dict, current_doc_id: str) -> None:
    with st.container(border=True):
        top = st.container(horizontal=True, vertical_alignment="center")
        top.caption(f"source_doc: `{chunk['source_doc']}`")
        with top:
            render_match_badge(chunk["source_doc"], current_doc_id)
        st.text(chunk["content"][:400])


def render_attribution_html(attributions: list[dict]) -> str:
    spans = []
    for a in attributions:
        text = html_lib.escape(a["sentence"])
        if a["source_doc"] is None:
            bg, border, label = "rgba(137,135,129,0.18)", "#898781", "unattributed (no close chunk match)"
        elif a["match"]:
            bg, border = "rgba(12,163,12,0.16)", STATUS_COLORS["good"]
            label = f"matches current document (similarity {a['similarity']})"
        else:
            bg, border = "rgba(208,59,59,0.16)", STATUS_COLORS["critical"]
            label = f"MISMATCH: from {a['source_doc']} (similarity {a['similarity']})"
        spans.append(
            f'<span title="{html_lib.escape(label)}" '
            f'style="background:{bg};border-bottom:3px solid {border};'
            f'padding:1px 4px;border-radius:3px;margin-right:3px;line-height:2.1;">{text}</span>'
        )
    return " ".join(spans)


def sidebar() -> None:
    with st.sidebar:
        header = st.container(horizontal=True, vertical_alignment="center")
        header.subheader("CRAIL", divider=False)
        header.badge("local", color="gray")

        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY is not set. Add it to a .env file.", icon=":material/key_off:")

        st.subheader("Documents", icon=":material/description:", divider="gray")

        clear_first = st.checkbox(
            "Clear existing index before adding (recommended)",
            value=True,
            help=(
                "Uncheck this to deliberately reproduce Phantom Context "
                "Hallucination: upload doc A, then upload doc B with this "
                "unchecked, and old chunks from A stay retrievable."
            ),
        )

        uploaded = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
        )

        if st.button("Ingest", disabled=not uploaded, icon=":material/upload_file:", type="primary", width="stretch"):
            if clear_first:
                try:
                    vectorstore.clear_index()
                    registry.clear_registry()
                except RuntimeError as e:
                    st.error(str(e), icon=":material/error:")
                    st.stop()
            for f in uploaded:
                with st.spinner(f"Ingesting {f.name}..."):
                    try:
                        docs = ingest_pdf(f.read(), doc_id=f.name)
                        vectorstore.add_documents(docs)
                        registry.register_document(f.name, len(docs), index_was_cleared=False)
                        st.success(f"Added {f.name} ({len(docs)} chunks)", icon=":material/check_circle:")
                    except Exception as e:
                        st.error(f"Failed on {f.name}: {e}", icon=":material/error:")
            # A new document changes what "the current question" even means,
            # so any question typed for the old context shouldn't survive.
            st.session_state["ask_query_input"] = ""
            st.rerun()

        known = registry.known_doc_ids()
        if known:
            st.caption(f"{len(known)} document{'s' if len(known) != 1 else ''} ever added to the index")
            for doc_id in known:
                st.badge(doc_id, icon=":material/description:", color="gray")
        else:
            st.caption("No documents indexed yet.")

        st.subheader("Detection settings", icon=":material/tune:", divider="gray")
        method_label = st.radio(
            "Confidence detection method",
            ["LLM judge, recommended (extra API call)", "Keyword matching"],
            help=(
                "Across 200 real evaluated queries, keyword matching agreed "
                "with the LLM judge on severity 0 times. Modern models "
                "answer factually without using any listed trigger phrase, "
                "so keyword matching is kept for comparison, not because "
                "it's reliable."
            ),
        )
        st.session_state.confidence_method = (
            "llm_judge" if method_label.startswith("LLM") else "keyword"
        )

        with st.expander("System info", icon=":material/info:"):
            st.caption(f"Chat model: `{CHAT_MODEL}`")
            st.caption(f"Judge model: `{JUDGE_MODEL}`")
            st.caption(f"Embedding model: `{EMBEDDING_MODEL}`")
            st.caption(f"Chunk size / overlap: {CHUNK_SIZE} / {CHUNK_OVERLAP}")
            st.caption(f"Top-k retrieval: {TOP_K}")
            st.caption(
                f"RRI weights: contamination {CONTAMINATION_WEIGHT}, "
                f"confidence {CONFIDENCE_WEIGHT}"
            )
            st.caption(
                f"Tier thresholds: Tier 2 ≥ {TIER2_THRESHOLD}, "
                f"Tier 3 ≥ {TIER3_THRESHOLD}"
            )
            st.caption(f"Confidence floor: contamination > {CONFIDENCE_MISMATCH_CONTAMINATION_FLOOR}")

        st.subheader("Danger zone", icon=":material/warning:", divider="gray")
        if st.button("Reset everything (index, queue, logs)", icon=":material/restart_alt:", width="stretch"):
            try:
                vectorstore.clear_index()
            except RuntimeError as e:
                st.error(str(e), icon=":material/error:")
                st.stop()
            registry.clear_registry()
            review_queue.clear_queue()
            audit.clear_log()
            st.rerun()


def status_bar(pending_total: int) -> None:
    known = registry.known_doc_ids()
    method = "LLM judge" if st.session_state.get("confidence_method") == "llm_judge" else "Keyword matching"

    with st.container(border=True):
        row = st.container(horizontal=True, vertical_alignment="center")
        row.badge(
            f"{len(known)} document{'s' if len(known) != 1 else ''} indexed",
            icon=":material/description:",
            color="blue",
        )
        if pending_total:
            row.badge(f"{pending_total} pending review", icon=":material/pending_actions:", color="orange")
        else:
            row.badge("review queue clear", icon=":material/check_circle:", color="green")
        row.badge(f"confidence: {method}", icon=":material/psychology:", color="gray")
        row.badge(f"Tier 2 ≥ {TIER2_THRESHOLD} · Tier 3 ≥ {TIER3_THRESHOLD}", color="gray")


def ask_tab() -> None:
    known = registry.known_doc_ids()
    if not known:
        with st.container(border=True, horizontal_alignment="center"):
            st.space("medium")
            st.subheader("No documents yet", icon=":material/upload_file:", text_alignment="center")
            st.caption(
                "Upload a PDF from the sidebar to start asking questions.",
                text_alignment="center",
            )
            st.space("medium")
        return

    current_doc_id = st.selectbox(
        "Which document are you currently asking about?",
        options=known,
        index=len(known) - 1,
        help="This is compared against each retrieved chunk's source_doc label.",
    )

    query = st.text_input("Ask a question", key="ask_query_input")
    submit = st.button("Ask", type="primary", icon=":material/send:", disabled=not query)

    if not submit:
        return

    vs = vectorstore.load_vectorstore()
    if vs is None:
        st.warning("Index is empty.")
        return

    client = get_client()
    intent = classify_intent(query)
    with st.spinner("Retrieving and generating..."):
        if intent == "specific":
            retrieved = retrieve(vs, query, k=TOP_K)
        else:
            retrieved = retrieve_broad(vs, query, current_doc_id, k=BROAD_TOP_K)
        if not retrieved:
            st.warning("Nothing retrieved for this query.")
            return
        answer = generate_answer(query, retrieved, client, intent=intent)
        decision = crail_decision(
            retrieved,
            answer,
            current_doc_id,
            confidence_method=st.session_state.get("confidence_method", "llm_judge"),
            client=client,
        )

    log_record = {
        "kind": "query_decision",
        "query": query,
        "current_doc_id": current_doc_id,
        "answer": answer,
        "intent": intent,
        **decision,
        "retrieved_sources": [d.metadata.get("source_doc") for d in retrieved],
    }
    audit.append_log(log_record)

    # Remember what was retrieved so the Embedding Space tab can circle
    # these points on the projection.
    st.session_state.last_retrieved_chunks = [
        {"content": d.page_content, "source_doc": d.metadata.get("source_doc")} for d in retrieved
    ]

    intent_icons = {"specific": ":material/search:", "summary": ":material/summarize:", "analysis": ":material/insights:"}
    st.caption(
        f"{intent_icons.get(intent, '')} Detected intent: **{intent}** "
        f"({len(retrieved)} chunks retrieved{', broad mode' if intent != 'specific' else ''})"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("RRI", decision["rri"], icon=":material/speed:", border=True)
    m2.metric("Contamination", f"{decision['contamination_ratio']:.0%}", icon=":material/blur_on:", border=True)
    m3.metric("Tier", f"{decision['tier']}: {decision['flag'].replace('_', ' ').title()}", icon=":material/layers:", border=True)

    naive_col, crail_col = st.columns(2)

    with naive_col:
        st.subheader("Without CRAIL", icon=":material/lock_open:")
        st.caption("What a pipeline with no safety layer would deliver, unconditionally.")
        st.write(answer)
        if decision["contamination_ratio"] > 0:
            st.caption(
                f":material/warning: Generated from context that was "
                f"{decision['contamination_ratio']:.0%} contaminated. A naive "
                f"pipeline ships this anyway."
            )

    with crail_col:
        st.subheader("With CRAIL", icon=":material/shield:")
        if decision["tier"] == 1:
            st.success(answer, icon=":material/check_circle:")
        elif decision["tier"] == 2:
            st.warning(
                "The answer is not shown to the user until a reviewer acts.",
                title="Held for human review (Tier 2, medium retrieval risk)",
                icon=":material/pending_actions:",
            )
            review_queue.add_item(
                {
                    "tier": 2,
                    "query": query,
                    "current_doc_id": current_doc_id,
                    "draft_answer": answer,
                    "rri": decision["rri"],
                    "contamination_ratio": decision["contamination_ratio"],
                    "confidence_mismatch": decision["confidence_mismatch"],
                    "retrieved": [
                        {
                            "content": d.page_content,
                            "source_doc": d.metadata.get("source_doc"),
                        }
                        for d in retrieved
                    ],
                }
            )
        else:
            st.error(
                "No automatic answer is delivered. A human must compose the "
                "reply from the Review Queue tab.",
                title="Blocked (Tier 3, high retrieval risk)",
                icon=":material/block:",
            )
            review_queue.add_item(
                {
                    "tier": 3,
                    "query": query,
                    "current_doc_id": current_doc_id,
                    "draft_answer": answer,
                    "rri": decision["rri"],
                    "contamination_ratio": decision["contamination_ratio"],
                    "confidence_mismatch": decision["confidence_mismatch"],
                    "retrieved": [
                        {
                            "content": d.page_content,
                            "source_doc": d.metadata.get("source_doc"),
                        }
                        for d in retrieved
                    ],
                }
            )

    with st.expander("Retrieval provenance", icon=":material/travel_explore:"):
        for doc in retrieved:
            render_chunk(doc, current_doc_id)

    with st.expander("Sentence-level attribution", icon=":material/format_color_text:"):
        st.caption(
            "Each sentence matched to its most similar retrieved chunk by "
            "embedding similarity, a best-effort visualization, not a formal "
            "citation. Green = traces to the current document, red = traces "
            "to a different (contaminating) document, gray = no close match."
        )
        attributions = attribute_sentences(
            answer, retrieved, current_doc_id, vectorstore.get_embeddings()
        )
        st.markdown(render_attribution_html(attributions), unsafe_allow_html=True)


def clean_chunks_of(item: dict) -> list[dict]:
    return [c for c in item["retrieved"] if c["source_doc"] == item["current_doc_id"]]


def grounding_excerpts_for(item: dict, vs) -> list[str]:
    """Verified content to check a human response against. Prefers the
    chunks already retrieved for this query; if none matched the current
    document (the worst-contamination case), falls back to a fresh search
    scoped only to that document's chunks, since the original top-k may
    have missed relevant content purely due to bad luck against the
    contaminating document."""
    clean = clean_chunks_of(item)
    if clean:
        return [c["content"] for c in clean]
    if vs is None:
        return []
    filtered = vs.similarity_search(
        item["query"], k=4, fetch_k=FILTER_FETCH_K, filter={"source_doc": item["current_doc_id"]}
    )
    return [d.page_content for d in filtered]


def review_tab() -> None:
    client = get_client()
    vs = vectorstore.load_vectorstore()

    reviewer = st.text_input(
        "Reviewer name (attached to every decision below, for an inter-rater audit trail)",
        key="reviewer_name",
    )

    st.subheader("Tier 2: hold for review", icon=":material/pending_actions:", divider="gray")
    tier2 = review_queue.pending_items(tier=2)
    if not tier2:
        st.caption("Nothing pending.")
    for item in tier2:
        clean = clean_chunks_of(item)
        scoped = retrieve_scoped(vs, item["query"], item["current_doc_id"], k=BROAD_TOP_K) if vs is not None else []
        with st.container(border=True):
            st.markdown(f"**Query:** {item['query']}")
            st.markdown(f"**Current document:** `{item['current_doc_id']}`")
            st.markdown(f"**Draft answer (from all retrieved chunks):** {item['draft_answer']}")
            st.caption(
                f"RRI {item['rri']} | contamination {item['contamination_ratio']} | "
                f"confidence mismatch: {item['confidence_mismatch']} | "
                f"{len(clean)}/{len(item['retrieved'])} originally-retrieved chunks matched "
                f"({len(scoped)} found in a fresh document-scoped search)"
            )
            with st.expander("Retrieved chunks", icon=":material/list:"):
                for chunk in item["retrieved"]:
                    render_chunk_dict(chunk, item["current_doc_id"])

            if item.get("suggested_answer"):
                st.info(f"**Suggested answer (matched chunks only):** {item['suggested_answer']}", icon=":material/auto_awesome:")

            if st.button(
                "Regenerate from matched chunks only",
                key=f"regen_{item['id']}",
                icon=":material/auto_awesome:",
                disabled=not scoped,
                help=None if scoped else "No content from the current document was found in the index.",
            ):
                suggestion = generate_answer_from_texts(
                    item["query"], [d.page_content for d in scoped], client
                )
                review_queue.patch_item(item["id"], suggested_answer=suggestion)
                st.rerun()

            c1, c2, c3 = st.columns(3)
            approve_text = item.get("suggested_answer") or item["draft_answer"]
            if c1.button("Approve as-is", key=f"approve_{item['id']}", icon=":material/check:", width="stretch"):
                review_queue.update_item(
                    item["id"],
                    status="approved",
                    final_answer=approve_text,
                    reviewer=reviewer,
                )
                st.rerun()
            if c2.button("Approve with note", key=f"note_{item['id']}", icon=":material/fact_check:", width="stretch"):
                note = (
                    "\n\n_Note: some retrieved context for this answer may not "
                    "be from the document you're currently asking about. "
                    "Treat this answer with caution._"
                )
                review_queue.update_item(
                    item["id"],
                    status="approved_with_note",
                    final_answer=approve_text + note,
                    reviewer=reviewer,
                )
                st.rerun()
            if c3.button("Escalate to Tier 3", key=f"escalate_{item['id']}", icon=":material/arrow_upward:", width="stretch"):
                review_queue.update_item(
                    item["id"], status="escalated", tier=3, reviewer=reviewer
                )
                st.rerun()

    st.subheader("Tier 3: blocked, needs human response", icon=":material/block:", divider="gray")
    tier3 = [i for i in review_queue.pending_items(tier=3)]
    if not tier3:
        st.caption("Nothing pending.")
    for item in tier3:
        clean = clean_chunks_of(item)
        scoped = retrieve_scoped(vs, item["query"], item["current_doc_id"], k=BROAD_TOP_K) if vs is not None else []
        text_key = f"human_{item['id']}"
        with st.container(border=True):
            st.markdown(f"**Query:** {item['query']}")
            st.markdown(f"**Current document:** `{item['current_doc_id']}`")
            st.caption(
                f"RRI {item['rri']} | contamination {item['contamination_ratio']} | "
                f"confidence mismatch: {item['confidence_mismatch']} | "
                f"{len(clean)}/{len(item['retrieved'])} originally-retrieved chunks matched "
                f"({len(scoped)} found in a fresh document-scoped search)"
            )
            with st.expander("Model's draft answer (not delivered, for reference only)", icon=":material/description:"):
                st.text(item["draft_answer"])
            with st.expander("Retrieved chunks", icon=":material/list:"):
                for chunk in item["retrieved"]:
                    render_chunk_dict(chunk, item["current_doc_id"])

            bcol1, bcol2 = st.columns(2)
            if bcol1.button(
                "Suggest answer from matched chunks only",
                key=f"suggest_{item['id']}",
                icon=":material/auto_awesome:",
                width="stretch",
                disabled=not scoped,
                help=None
                if scoped
                else "No content from the current document was found in the index, nothing trustworthy to suggest from.",
            ):
                suggestion = generate_answer_from_texts(
                    item["query"], [d.page_content for d in scoped], client
                )
                review_queue.patch_item(item["id"], suggested_answer=suggestion)
                st.session_state[text_key] = suggestion
                st.rerun()
            if bcol2.button("Clear draft", key=f"clear_{item['id']}", icon=":material/backspace:", width="stretch"):
                st.session_state[text_key] = ""
                st.rerun()

            text_area_kwargs = {"key": text_key}
            if text_key not in st.session_state:
                text_area_kwargs["value"] = item.get("suggested_answer", "")
            st.text_area(
                "Human response (edit freely, or submit the suggestion as-is)",
                **text_area_kwargs,
            )

            override_key = f"override_{item['id']}"
            override = st.checkbox(
                "Override grounding check (I've manually verified this against the source document)",
                key=override_key,
            )

            if st.button("Submit human response", key=f"submit_{item['id']}", icon=":material/send:", type="primary"):
                response_text = st.session_state[text_key]
                if override:
                    review_queue.update_item(
                        item["id"],
                        status="blocked_resolved",
                        final_answer=response_text,
                        grounding_override=True,
                        reviewer=reviewer,
                    )
                    audit.append_log(
                        {
                            "kind": "grounding_override",
                            "item_id": item["id"],
                            "query": item["query"],
                            "current_doc_id": item["current_doc_id"],
                            "final_answer": response_text,
                            "reviewer": reviewer,
                        }
                    )
                    st.rerun()
                else:
                    excerpts = grounding_excerpts_for(item, vs)
                    grounded, reason = check_grounding(response_text, excerpts, client)
                    if grounded:
                        review_queue.update_item(
                            item["id"],
                            status="blocked_resolved",
                            final_answer=response_text,
                            grounding_override=False,
                            reviewer=reviewer,
                        )
                        st.rerun()
                    else:
                        audit.append_log(
                            {
                                "kind": "grounding_rejection",
                                "item_id": item["id"],
                                "query": item["query"],
                                "current_doc_id": item["current_doc_id"],
                                "attempted_response": response_text,
                                "reason": reason,
                                "reviewer": reviewer,
                            }
                        )
                        st.error(reason, icon=":material/block:", title="Rejected")
                        st.caption(
                            "Revise the response so it's supported by the document, "
                            "or check the override box above if you've verified it "
                            "manually (for example against a source CRAIL can't see)."
                        )
                        if excerpts:
                            with st.expander("Checked against these excerpts", icon=":material/search:"):
                                for e in excerpts:
                                    st.text(e[:400])

    st.subheader("Resolved", icon=":material/history:", divider="gray")
    resolved = review_queue.resolved_items()
    if not resolved:
        st.caption("Nothing resolved yet.")
    for item in resolved:
        with st.expander(f"[{item['status']}] {item['query']}"):
            st.caption(
                f"Tier {item['tier']} at the time | RRI {item['rri']} | "
                f"contamination {item['contamination_ratio']} | "
                f"confidence mismatch: {item['confidence_mismatch']}"
            )

            naive_col, final_col = st.columns(2)
            with naive_col:
                st.markdown("**Without CRAIL** (naive draft, from all retrieved chunks)")
                st.caption(item["draft_answer"])
            with final_col:
                st.markdown("**After human review**")
                st.caption(item.get("final_answer") or "(no final answer recorded)")

            status_notes = []
            if item.get("grounding_override"):
                status_notes.append(":material/warning: submitted via grounding-check override")
            if item.get("suggested_answer") and item.get("suggested_answer") != item.get("final_answer"):
                status_notes.append("the CRAIL-suggested answer was edited before submission")
            if status_notes:
                st.caption(" · ".join(status_notes))

            st.caption(
                f"Status: `{item['status']}` | Reviewer: {item.get('reviewer') or '(unspecified)'} | "
                f"Resolved: {item.get('resolved_at', '(unknown)')}"
            )

            with st.expander("Retrieved chunks at the time", icon=":material/list:"):
                for chunk in item["retrieved"]:
                    render_chunk_dict(chunk, item["current_doc_id"])


def audit_tab() -> None:
    all_records = audit.read_log()
    records = [r for r in all_records if r.get("kind", "query_decision") == "query_decision"]
    rejections = [r for r in all_records if r.get("kind") == "grounding_rejection"]
    overrides = [r for r in all_records if r.get("kind") == "grounding_override"]

    if not records:
        st.caption("No queries logged yet.")
        return

    total = len(records)
    tier_counts = {1: 0, 2: 0, 3: 0}
    for r in records:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
    intercepted = tier_counts.get(2, 0) + tier_counts.get(3, 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queries", total, icon=":material/forum:", border=True)
    c2.metric("Tier 1, delivered", tier_counts.get(1, 0), icon=":material/check_circle:", border=True)
    c3.metric("Tier 2, held", tier_counts.get(2, 0), icon=":material/pending_actions:", border=True)
    c4.metric("Tier 3, blocked", tier_counts.get(3, 0), icon=":material/block:", border=True)
    st.metric("Interception rate", f"{intercepted / total:.0%}", icon=":material/shield:", border=True)

    st.caption(
        f"Thresholds: Tier 2 at RRI ≥ {TIER2_THRESHOLD}, Tier 3 at RRI ≥ {TIER3_THRESHOLD}"
    )
    st.dataframe(
        [
            {
                "query": r["query"],
                "tier": r["tier"],
                "rri": r["rri"],
                "contamination": r["contamination_ratio"],
                "confidence_mismatch": r["confidence_mismatch"],
                "current_doc": r["current_doc_id"],
                "logged_at": r["logged_at"],
            }
            for r in reversed(records)
        ],
        width="stretch",
    )

    st.subheader("Human-response grounding checks", icon=":material/verified_user:", divider="gray")
    st.caption("The Tier 3 security layer: checks on human-composed responses before they're delivered.")
    gc1, gc2 = st.columns(2)
    gc1.metric("Rejected, unsupported response", len(rejections), icon=":material/gpp_bad:", border=True)
    gc2.metric("Manually overridden", len(overrides), icon=":material/admin_panel_settings:", border=True)
    if rejections:
        with st.expander("Rejected submissions", icon=":material/gpp_bad:"):
            for r in reversed(rejections):
                st.markdown(f"**Query:** {r['query']}")
                st.text(f"Attempted: {r['attempted_response'][:300]}")
                st.caption(f"Reason: {r['reason']} | {r['logged_at']}")
                st.divider()
    if overrides:
        with st.expander("Overridden submissions", icon=":material/admin_panel_settings:"):
            for r in reversed(overrides):
                st.markdown(f"**Query:** {r['query']}")
                st.text(f"Final answer: {r['final_answer'][:300]}")
                st.caption(r["logged_at"])
                st.divider()


def viz_tab() -> None:
    st.subheader("Embedding space", icon=":material/scatter_plot:")
    st.caption(
        "2D PCA projection of every chunk currently in the FAISS index, "
        "colored by source document. Black-ringed markers were retrieved "
        "for your most recent query in the Ask tab."
    )

    vs = vectorstore.load_vectorstore()
    if vs is None:
        st.info("Index is empty. Upload documents first.")
        return

    docs, vectors = get_all_chunks_with_vectors(vs)
    if len(docs) < 2:
        st.info("Need at least 2 chunks in the index to project.")
        return

    coords = project_2d(vectors)
    last_retrieved_content = {
        c["content"] for c in st.session_state.get("last_retrieved_chunks", [])
    }

    doc_ids = sorted({d.metadata.get("source_doc", "UNKNOWN") for d in docs})
    color_map = {doc_id: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, doc_id in enumerate(doc_ids)}

    fig = go.Figure()
    for doc_id in doc_ids:
        idxs = [i for i, d in enumerate(docs) if d.metadata.get("source_doc", "UNKNOWN") == doc_id]
        retrieved_flags = [docs[i].page_content in last_retrieved_content for i in idxs]
        fig.add_trace(
            go.Scatter(
                x=[coords[i, 0] for i in idxs],
                y=[coords[i, 1] for i in idxs],
                mode="markers",
                name=doc_id,
                marker=dict(
                    size=14,
                    color=color_map[doc_id],
                    line=dict(
                        width=[3 if r else 0 for r in retrieved_flags],
                        color="#0b0b0b",
                    ),
                ),
                text=[docs[i].page_content[:180] + "..." for i in idxs],
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        legend_title_text="Source document",
        xaxis_title="PC1",
        yaxis_title="PC2",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        margin=dict(l=10, r=10, t=30, b=10),
        height=520,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Two documents whose chunks cluster close together (or interleave) "
        "in this space are exactly the high-risk pair type: MMR retrieval "
        "is far more likely to pull a wrong-document chunk into the top-k "
        "when the two documents are semantically similar."
    )


def sensitivity_tab() -> None:
    st.subheader("Threshold and weight sensitivity", icon=":material/tune:")
    st.caption(
        "Replays the RRI formula over a saved evaluate.py run under "
        "different hyperparameters. Pure arithmetic on already-collected "
        "data, no new API calls."
    )

    results_path = Path("eval_results.json")
    if not results_path.exists():
        st.info(
            "No evaluation data found. Run `evaluate.py` from the project "
            "root first to generate eval_results.json.",
            icon=":material/dataset:",
        )
        return

    results = json.loads(results_path.read_text(encoding="utf-8"))
    n_contam = sum(1 for r in results if r["scenario"] == "contaminated")
    n_clean = sum(1 for r in results if r["scenario"] == "clean")
    st.caption(f"Loaded {len(results)} queries ({n_contam} contaminated, {n_clean} clean) from eval_results.json.")

    st.markdown("##### Tier 2 threshold sweep")
    thresholds = [round(x, 2) for x in np.arange(0.05, 0.85, 0.05)]
    rows = sweep_tier2_threshold(results, thresholds)
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=[r["threshold"] for r in rows],
            y=[r["interception_rate"] for r in rows],
            mode="lines+markers",
            name="Interception rate",
            line=dict(color=STATUS_COLORS["good"], width=2),
            marker=dict(size=7),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=[r["threshold"] for r in rows],
            y=[r["false_positive_rate"] for r in rows],
            mode="lines+markers",
            name="False positive rate",
            line=dict(color=STATUS_COLORS["critical"], width=2),
            marker=dict(size=7),
        )
    )
    fig1.add_vline(
        x=TIER2_THRESHOLD,
        line_dash="dash",
        line_color="#898781",
        annotation_text=f"default ({TIER2_THRESHOLD})",
        annotation_position="top",
    )
    fig1.update_layout(
        xaxis_title="Tier 2 threshold",
        yaxis_title="Rate",
        yaxis_range=[0, 1.05],
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        legend_title_text="",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig1, width="stretch")

    st.markdown("##### Contamination weight sweep (Tier 2 threshold fixed at default)")
    weights = [round(x, 2) for x in np.arange(0.1, 1.05, 0.1)]
    rows2 = sweep_contamination_weight(results, weights)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=[r["contamination_weight"] for r in rows2],
            y=[r["interception_rate"] for r in rows2],
            mode="lines+markers",
            name="Interception rate",
            line=dict(color=STATUS_COLORS["good"], width=2),
            marker=dict(size=7),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[r["contamination_weight"] for r in rows2],
            y=[r["false_positive_rate"] for r in rows2],
            mode="lines+markers",
            name="False positive rate",
            line=dict(color=STATUS_COLORS["critical"], width=2),
            marker=dict(size=7),
        )
    )
    fig2.add_vline(
        x=CONTAMINATION_WEIGHT,
        line_dash="dash",
        line_color="#898781",
        annotation_text=f"default ({CONTAMINATION_WEIGHT})",
        annotation_position="top",
    )
    fig2.update_layout(
        xaxis_title="Contamination weight",
        yaxis_title="Rate",
        yaxis_range=[0, 1.05],
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        legend_title_text="",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig2, width="stretch")

    st.caption(
        "Both sweeps replay the fixed evaluation set from evaluate.py (3 "
        "document pairs, 30 queries). Re-run evaluate.py with more or "
        "different document pairs before treating these curves as general: "
        "interception rate varies by document-pair type (see the README)."
    )


st.title("CRAIL: Contextual Risk-Aware Intervention Layer", icon=":material/shield:")
st.caption(
    "A human-in-the-loop safety layer that catches Phantom Context "
    "Hallucination: stale chunks from a previous document silently "
    "contaminating an answer about the current one."
)

sidebar()

_pending_total = len(review_queue.pending_items(tier=2)) + len(review_queue.pending_items(tier=3))
status_bar(_pending_total)

# Tab labels must be static: `default` below has to match one exactly to
# restore the active tab after a rerun, and a count that changes as items
# get resolved (e.g. "Review queue (3)" -> "Review queue (2)") would break
# that match. The pending count is already visible in the status bar above,
# so it isn't lost by keeping these labels fixed.
_TAB_LABELS = [
    ":material/forum: Ask",
    ":material/fact_check: Review queue",
    ":material/history: Audit log",
    ":material/scatter_plot: Embedding space",
    ":material/tune: Sensitivity analysis",
]
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    _TAB_LABELS,
    key="main_tabs",
    default=st.session_state.get("main_tabs", _TAB_LABELS[0]),
    on_change="rerun",
)
with tab1:
    ask_tab()
with tab2:
    review_tab()
with tab3:
    audit_tab()
with tab4:
    viz_tab()
with tab5:
    sensitivity_tab()
