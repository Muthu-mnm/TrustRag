"""
TrustRAG: Hallucination Detection and Abstention Calibration in Large Language Models
Using Evidence-Grounded Question Answering.

Clean, Modern, ChatGPT-style Research Document Assistant.
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np

from trustrag.config import (
    DEFAULT_LLM_MODEL,
    FALLBACK_LLM_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_LLM_HOST,
    DEFAULT_OLLAMA_EMBED_HOST,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_ABSTENTION_THRESHOLD,
    DEFAULT_SUFFICIENCY_WEIGHT,
    DEFAULT_RETRIEVAL_WEIGHT,
    DEFAULT_PIPELINE_MODE,
    ABSTENTION_MESSAGE,
)
from trustrag.ollama_client import OllamaClient
from trustrag.pdf_processor import process_pdf_file
from trustrag.vector_store import VectorStoreManager
from trustrag.generator import RAGPipeline
from trustrag.evaluator import EvaluationEngine
from trustrag.visualization import (
    create_metric_comparison_chart,
    create_reliability_diagram,
    create_risk_coverage_chart,
)

# Page configuration
st.set_page_config(
    page_title="TrustRAG • Evidence-Grounded Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent modern dark theme styling
st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .top-brand-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0px 14px 0px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 12px;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-pill {
        background: #1e293b;
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 9999px;
        border: 1px solid #334155;
    }
    
    /* Document Badge Bar */
    .doc-pill-card {
        background: #131d2e;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 10px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }
    
    /* Chat Message Cards & Badges */
    .stChatMessage {
        background-color: transparent !important;
        padding: 8px 0px !important;
    }
    
    .badge-pass {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #059669;
        display: inline-block;
    }
    .badge-abstain {
        background-color: #450a0a;
        color: #fca5a5;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #dc2626;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #1e293b;
        color: #94a3b8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid #334155;
        display: inline-block;
    }
    .badge-info {
        background-color: #0c4a6e;
        color: #7dd3fc;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #0284c7;
        display: inline-block;
    }

    /* Refusal Callout Box */
    .refusal-box {
        background-color: #1a1016;
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .refusal-title {
        color: #f87171;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    
    /* Verified Answer Box */
    .verified-box {
        background-color: #0d1a24;
        border-left: 4px solid #10b981;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    /* Diagnostics Row */
    .diag-pills-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 8px;
        margin-bottom: 6px;
    }
    
    /* Empty State Container */
    .empty-state {
        text-align: center;
        padding: 40px 20px;
        background: #111827;
        border: 1px dashed #374151;
        border-radius: 12px;
        margin: 20px 0;
    }
    .empty-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 6px;
    }
    .empty-desc {
        color: #94a3b8;
        font-size: 0.9rem;
        max-width: 500px;
        margin: 0 auto 16px auto;
    }

    /* Sidebar styling for ChatGPT-style history */
    section[data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid #1e293b;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vector_store_manager" not in st.session_state:
    st.session_state["vector_store_manager"] = None
if "indexed_filename" not in st.session_state:
    st.session_state["indexed_filename"] = None
if "doc_stats" not in st.session_state:
    st.session_state["doc_stats"] = {}
if "eval_results" not in st.session_state:
    st.session_state["eval_results"] = None
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None

# --- Backend Configuration (Configured strictly in backend / config.py) ---
selected_pipeline_mode = DEFAULT_PIPELINE_MODE
selected_model = DEFAULT_LLM_MODEL
embed_model = DEFAULT_EMBED_MODEL
llm_host = DEFAULT_OLLAMA_LLM_HOST
embed_host = DEFAULT_OLLAMA_EMBED_HOST
chunk_size = DEFAULT_CHUNK_SIZE
chunk_overlap = DEFAULT_CHUNK_OVERLAP
top_k = DEFAULT_TOP_K
abstention_threshold = DEFAULT_ABSTENTION_THRESHOLD
weight_suff = DEFAULT_SUFFICIENCY_WEIGHT
weight_ret = DEFAULT_RETRIEVAL_WEIGHT

# Instantiate Ollama client
llm_client = OllamaClient(base_url=llm_host, timeout=120)

# --- Sidebar: ChatGPT-Style Chat History ---
with st.sidebar:
    st.markdown("### 💬 Chat History")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state["messages"] = []
        st.session_state["pending_query"] = None
        st.rerun()

    st.markdown("<hr style='border: none; border-top: 1px solid #1e293b; margin: 10px 0;'>", unsafe_allow_html=True)
    
    # Collect past user queries
    user_queries = [
        (idx, m["content"])
        for idx, m in enumerate(st.session_state.get("messages", []))
        if m["role"] == "user"
    ]

    if not user_queries:
        st.caption("No conversations in this session yet.")
    else:
        st.caption(f"{len(user_queries)} question(s) asked:")
        for q_idx, (m_idx, query_text) in enumerate(reversed(user_queries)):
            display_text = query_text if len(query_text) <= 32 else query_text[:29] + "..."
            if st.button(f"🗨️ {display_text}", key=f"hist_btn_{m_idx}", use_container_width=True):
                st.session_state["pending_query"] = query_text
                st.rerun()

    # Bottom Clear History button
    if user_queries:
        st.markdown("<hr style='border: none; border-top: 1px solid #1e293b; margin: 12px 0;'>", unsafe_allow_html=True)
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    st.markdown(
        """
        <div style="margin-top: 24px; padding: 10px; background: #131d2e; border: 1px solid #1e293b; border-radius: 6px; font-size: 0.75rem; color: #94a3b8;">
            <div style="font-weight: 600; color: #38bdf8; margin-bottom: 2px;">🛡️ TrustRAG Engine</div>
            <div>Model: <b>gemma3:4b (Local)</b></div>
            <div>Embeddings: <b>nomic-embed-text (Local)</b></div>
            <div>Calibrated Abstention: <b>Active</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Top Navigation Bar & Brand Header ---
col_logo, col_nav = st.columns([1.5, 2.5])

with col_logo:
    st.markdown(
        """
        <div class="top-brand-bar">
            <div class="brand-title">
                <span>🛡️ TrustRAG</span>
                <span class="brand-pill">Evidence-Grounded Assistant</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_nav:
    # Modern horizontal segmented navigation
    nav_view = st.radio(
        "Navigation",
        options=["💬 Document Chat", "📊 Evaluation"],
        horizontal=True,
        label_visibility="collapsed",
    )

# Collapsible About row to keep the initial viewport compact
with st.expander("ℹ️ About TrustRAG (Problem Formulation & Calibrated Abstention)", expanded=False):
    st.markdown(
        """
        **TrustRAG** tackles LLM hallucination in Retrieval-Augmented Generation through:
        1. **Pre-generation Evidence Sufficiency Gating:** Verifies whether retrieved passages contain all material facts.
        2. **Composite Confidence Estimation:** Combines LLM self-reported sufficiency and normalized retrieval similarity.
        3. **Calibrated Abstention:** Transparently refuses out-of-scope or misleading questions instead of guessing.
        4. **Post-generation Factual Verification:** Validates generated claims against evidence before display.
        """
    )

# ==============================================================================
# VIEW 1: Document Chat (Default Workspace)
# ==============================================================================
if nav_view == "💬 Document Chat":

    # Active Document Banner / Compact Attachment Bar
    if st.session_state["vector_store_manager"] is not None:
        stats = st.session_state["doc_stats"]
        doc_col1, doc_col2 = st.columns([4, 1])
        with doc_col1:
            st.markdown(
                f"""
                <div class="doc-pill-card">
                    <div>
                        <span style="font-weight: 600; color: #f8fafc;">📄 {stats.get('filename')}</span>
                        <span style="color: #64748b; margin: 0 8px;">•</span>
                        <span style="color: #94a3b8; font-size: 0.88rem;">{stats.get('page_count')} pages</span>
                        <span style="color: #64748b; margin: 0 8px;">•</span>
                        <span style="color: #38bdf8; font-size: 0.88rem;">{stats.get('chunk_count')} chunks indexed</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with doc_col2:
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("🔄 Replace", use_container_width=True):
                    st.session_state["vector_store_manager"] = None
                    st.session_state["indexed_filename"] = None
                    st.session_state["doc_stats"] = {}
                    st.session_state["messages"] = []
                    st.rerun()
            with bc2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state["messages"] = []
                    st.rerun()

    else:
        # Compact PDF Dropzone when no document is active
        st.markdown(
            """
            <div style="background: #111827; border: 1px dashed #374151; border-radius: 8px; padding: 16px 20px; margin-bottom: 12px;">
                <div style="font-weight: 600; color: #f1f5f9; font-size: 0.95rem; margin-bottom: 2px;">📄 Upload Knowledge Base PDF</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">Upload a research paper, documentation, or technical report to activate evidence-grounded QA.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
            help="Select a PDF file with selectable text",
        )

        local_sample_pdf = os.path.join(os.path.dirname(__file__), "Inception net.pdf")
        if os.path.exists(local_sample_pdf):
            st.markdown("<div style='text-align: center; margin: 4px 0 6px 0; color: #64748b; font-size: 0.8rem;'>— OR —</div>", unsafe_allow_html=True)
            if st.button("📄 Load 'Inception net.pdf' (Workspace Sample)", use_container_width=True):
                with open(local_sample_pdf, "rb") as f:
                    file_bytes = f.read()
                file_name = "Inception net.pdf"
                with st.spinner("Extracting page metadata and building FAISS vector index..."):
                    proc_res = process_pdf_file(
                        file_bytes=file_bytes,
                        filename=file_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    if proc_res["error"]:
                        st.error(f"PDF Processing Failed: {proc_res['error']}")
                    elif not proc_res["documents"]:
                        st.warning("No selectable text extracted.")
                    else:
                        vsm = VectorStoreManager(embed_model=embed_model, embed_base_url=embed_host)
                        vsm.build_index(proc_res["documents"])
                        st.session_state["vector_store_manager"] = vsm
                        st.session_state["indexed_filename"] = file_name
                        st.session_state["doc_stats"] = {
                            "filename": file_name,
                            "page_count": proc_res["page_count"],
                            "chunk_count": proc_res["chunk_count"],
                        }
                        st.session_state["messages"] = []
                        st.rerun()

        if uploaded_pdf is not None:
            file_bytes = uploaded_pdf.read()
            file_name = uploaded_pdf.name

            with st.spinner("Extracting page metadata and building FAISS vector index..."):
                proc_res = process_pdf_file(
                    file_bytes=file_bytes,
                    filename=file_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                if proc_res["error"]:
                    st.error(f"PDF Processing Failed: {proc_res['error']}")
                elif not proc_res["documents"]:
                    st.warning("No selectable text extracted. Please ensure the document is not an image-only scan.")
                else:
                    vsm = VectorStoreManager(
                        embed_model=embed_model,
                        embed_base_url=embed_host,
                    )
                    try:
                        vsm.build_index(proc_res["documents"])
                        st.session_state["vector_store_manager"] = vsm
                        st.session_state["indexed_filename"] = file_name
                        st.session_state["doc_stats"] = {
                            "filename": file_name,
                            "page_count": proc_res["page_count"],
                            "chunk_count": proc_res["chunk_count"],
                        }
                        st.session_state["messages"] = []
                        st.success(f"Indexed '{file_name}' ({proc_res['page_count']} pages, {proc_res['chunk_count']} chunks).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Vector Indexing Error: {str(e)}")

    # Conversation Display
    vsm = st.session_state["vector_store_manager"]

    if vsm is None:
        # Helpful Onboarding Empty State
        st.markdown(
            """
            <div class="empty-state">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🛡️</div>
                <div class="empty-title">Evidence-Grounded Question Answering</div>
                <div class="empty-desc">
                    Upload a PDF document above. TrustRAG will extract page citations, assess evidence sufficiency, 
                    estimate confidence, and abstain when evidence is incomplete or misleading.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # If no messages yet, show suggestion chips
        if len(st.session_state["messages"]) == 0:
            st.markdown(
                """
                <div style="margin: 24px 0 16px 0; color: #94a3b8; font-size: 0.9rem;">
                    💡 <b>Sample questions to test TrustRAG's calibration & abstention:</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                if st.button("📌 What is the main contribution?", use_container_width=True):
                    st.session_state["pending_query"] = "What is the main architecture or contribution described in this document?"
                    st.rerun()
            with col_s2:
                if st.button("📊 What experimental results are reported?", use_container_width=True):
                    st.session_state["pending_query"] = "What are the primary benchmark results or quantitative metrics reported?"
                    st.rerun()
            with col_s3:
                if st.button("⚠️ Test Abstention: Out-of-scope query", use_container_width=True):
                    st.session_state["pending_query"] = "What was the exact GPU operating temperature and monetary cloud cost during training?"
                    st.rerun()

        # Render Past Messages
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["content"])
                else:
                    data = msg.get("data", {})
                    mode_used = data.get("mode", "TrustRAG")

                    if mode_used == "Side-by-Side Comparison":
                        col_b, col_t = st.columns(2)
                        with col_b:
                            st.markdown("#### 🏛️ Baseline RAG")
                            st.write(data["base_answer"])
                            st.caption(f"Latency: {data['base_latency']}s | Gate: None")
                        with col_t:
                            st.markdown("#### 🛡️ TrustRAG (Ours)")
                            if data["trust_abstained"]:
                                st.markdown(
                                    f"""
                                    <div class="refusal-box">
                                        <div class="refusal-title">🛑 Refused: Insufficient Evidence</div>
                                        <div style="color: #cbd5e1; font-style: italic;">{data['trust_answer']}</div>
                                        <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 6px;"><b>Reason:</b> {data['trust_reason']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div class="verified-box">
                                        <div style="color: #e2e8f0;">{data['trust_answer']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            st.caption(f"Latency: {data['trust_latency']}s | Confidence: {data.get('trust_conf', 0.0):.2f}")

                    elif data.get("abstained"):
                        # Refusal / Abstention Message
                        st.markdown(
                            f"""
                            <div class="refusal-box">
                                <div class="refusal-title">🛑 Refused: Insufficient Document Evidence</div>
                                <div style="color: #cbd5e1; font-style: italic; margin-bottom: 6px;">
                                    "{data['answer']}"
                                </div>
                                <div style="font-size: 0.82rem; color: #94a3b8;">
                                    <b>Abstention Reason:</b> {data.get('abstention_reason', 'Evidence gate rejected query.')}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # Grounded Answer
                        st.markdown(
                            f"""
                            <div class="verified-box">
                                <div style="color: #f1f5f9; line-height: 1.6;">
                                    {data['answer']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Diagnostic metadata bar
                    conf_info = data.get("confidence_details") or {}
                    suff_info = data.get("sufficiency_assessment") or {}
                    verif_info = data.get("verification") or {}

                    if mode_used != "Side-by-Side Comparison":
                        diag_html = ['<div class="diag-pills-row">']
                        if conf_info:
                            c_val = conf_info.get("combined_confidence", 0.0)
                            diag_html.append(f'<span class="badge-info">Confidence: {c_val:.2f}</span>')
                        if suff_info:
                            s_lvl = suff_info.get("sufficiency_level", "Unknown").title()
                            diag_html.append(f'<span class="badge-neutral">Sufficiency: {s_lvl}</span>')
                        if verif_info:
                            v_ok = verif_info.get("supported", False)
                            badge_cls = "badge-pass" if v_ok else "badge-abstain"
                            v_label = "Verified Grounded" if v_ok else "Verification Failed"
                            diag_html.append(f'<span class="{badge_cls}">{v_label}</span>')
                        if data.get("latency_seconds"):
                            diag_html.append(f'<span class="badge-neutral">⏱️ {data["latency_seconds"]}s</span>')
                        diag_html.append('</div>')
                        st.markdown("".join(diag_html), unsafe_allow_html=True)

                        # Detailed Collapsible Trace
                        with st.expander("🔍 Evidence & Verification Trace", expanded=False):
                            if suff_info:
                                st.markdown("##### 1. Evidence Sufficiency Assessment")
                                st.write(f"- **Sufficient:** `{suff_info.get('sufficient')}`")
                                st.write(f"- **Self-Reported LLM Confidence:** `{suff_info.get('confidence')}`")
                                st.write(f"- **Auditor Reason:** {suff_info.get('reason')}")
                                if suff_info.get("missing_information"):
                                    st.write(f"- **Missing Information:** {suff_info.get('missing_information')}")

                            if conf_info:
                                st.markdown("##### 2. Confidence Formula Breakdown")
                                st.write(
                                    f"- **Formula:** `({conf_info.get('weights', {}).get('w_sufficiency')} × S_suff) + "
                                    f"({conf_info.get('weights', {}).get('w_retrieval')} × R_ret) = {conf_info.get('combined_confidence')}`\n"
                                    f"- **Abstention Threshold (τ):** `{conf_info.get('abstention_threshold')}`"
                                )

                            if verif_info:
                                st.markdown("##### 3. Answer-Support Verifier")
                                st.write(f"- **Claims Supported:** `{verif_info.get('supported')}`")
                                st.write(f"- **Verifier Confidence:** `{verif_info.get('confidence')}`")
                                st.write(f"- **Verifier Rationale:** {verif_info.get('reason')}")

                            passages = data.get("retrieved_passages", [])
                            if passages:
                                st.markdown(f"##### 4. Retrieved Evidence Passages ({len(passages)} passages)")
                                for p_idx, p in enumerate(passages, 1):
                                    st.markdown(
                                        f"**[Passage {p_idx}] Page {p['page']}** | Relevance: `{p['similarity_score']}`"
                                    )
                                    st.code(p["content"], language="text")

        # Chat Input Bar
        chat_prompt = st.chat_input("Ask a question about the document...")
        
        # Check if there is a pending suggestion click
        if st.session_state["pending_query"] is not None:
            chat_prompt = st.session_state["pending_query"]
            st.session_state["pending_query"] = None

        if chat_prompt:
            # 1. Append User Message
            st.session_state["messages"].append({"role": "user", "content": chat_prompt})

            # 2. Run Pipeline
            pipeline = RAGPipeline(
                ollama_client=llm_client,
                vector_store_manager=vsm,
                model_name=selected_model,
                abstention_threshold=abstention_threshold,
            )

            with st.spinner(f"Evaluating evidence with {selected_model}..."):
                if selected_pipeline_mode == "Side-by-Side Comparison":
                    base_out = pipeline.run_baseline_rag(chat_prompt, top_k=top_k)
                    trust_out = pipeline.run_trust_rag(chat_prompt, top_k=top_k)

                    conf_val = (
                        trust_out["confidence_details"]["combined_confidence"]
                        if trust_out.get("confidence_details")
                        else 0.0
                    )
                    assistant_data = {
                        "mode": "Side-by-Side Comparison",
                        "base_answer": base_out["answer"],
                        "base_latency": base_out["latency_seconds"],
                        "trust_answer": trust_out["answer"],
                        "trust_abstained": trust_out["abstained"],
                        "trust_reason": trust_out["abstention_reason"],
                        "trust_latency": trust_out["latency_seconds"],
                        "trust_conf": conf_val,
                    }
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": trust_out["answer"],
                        "data": assistant_data,
                    })
                elif selected_pipeline_mode == "Baseline RAG (Standard)":
                    base_out = pipeline.run_baseline_rag(chat_prompt, top_k=top_k)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": base_out["answer"],
                        "data": base_out,
                    })
                else:  # TrustRAG
                    trust_out = pipeline.run_trust_rag(chat_prompt, top_k=top_k)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": trust_out["answer"],
                        "data": trust_out,
                    })

            st.rerun()

# ==============================================================================
# VIEW 2: Evaluation & Calibration Dashboard
# ==============================================================================
elif nav_view == "📊 Evaluation":
    st.subheader("📊 Comparative Benchmark & Calibration Evaluation")
    st.markdown(
        "Evaluate Baseline RAG vs. TrustRAG against labelled datasets containing "
        "**Answerable**, **Unanswerable**, and **Misleading** test questions."
    )

    eval_col1, eval_col2 = st.columns([2, 1])

    with eval_col1:
        eval_csv = st.file_uploader("Upload Evaluation CSV:", type=["csv"])
    with eval_col2:
        st.write("**Or load reference dataset:**")
        load_sample = st.button("📁 Load Built-in Benchmark (10 Qs)", use_container_width=True)

    sample_csv_path = os.path.join(os.path.dirname(__file__), "data", "sample_evaluation.csv")
    df_eval = None

    if eval_csv is not None:
        try:
            df_eval = pd.read_csv(eval_csv)
            st.success(f"Loaded uploaded evaluation dataset with {len(df_eval)} questions.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif load_sample and os.path.exists(sample_csv_path):
        df_eval = pd.read_csv(sample_csv_path)
        st.success(f"Loaded built-in reference dataset ({len(df_eval)} questions).")

    if df_eval is not None:
        st.markdown("##### Benchmark Questions Preview")
        st.dataframe(df_eval[["question", "expected_answer", "answerable", "question_type"]], use_container_width=True)

        if st.session_state["vector_store_manager"] is None:
            st.warning("⚠️ Please upload a target PDF in Chat first to index the knowledge base before running the evaluation.")
        else:
            if st.button("⚡ Run Full Benchmark Evaluation", type="primary"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                vsm = st.session_state["vector_store_manager"]
                pipeline = RAGPipeline(
                    ollama_client=llm_client,
                    vector_store_manager=vsm,
                    model_name=selected_model,
                    abstention_threshold=abstention_threshold,
                )
                engine = EvaluationEngine(pipeline)

                def update_progress(current, total):
                    pct = current / total
                    progress_bar.progress(pct)
                    status_text.text(f"Evaluating Question {current}/{total} across both pipelines...")

                start_eval = time.time()
                with st.spinner("Executing benchmark across Baseline RAG and TrustRAG..."):
                    results = engine.evaluate_dataset(df_eval, progress_callback=update_progress)
                eval_duration = round(time.time() - start_eval, 1)

                st.session_state["eval_results"] = results
                progress_bar.empty()
                status_text.empty()
                st.success(f"Benchmark completed in {eval_duration} seconds!")

    # Display evaluation results if available
    if st.session_state.get("eval_results"):
        res = st.session_state["eval_results"]
        b_met = res["baseline_metrics"]
        t_met = res["trustrag_metrics"]

        st.divider()
        st.markdown("### 🏆 Core Metric Comparison Cards")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            acc_diff = (t_met["accuracy"] - b_met["accuracy"]) * 100
            st.metric(
                "Answer Accuracy",
                f"{t_met['accuracy']*100:.1f}%",
                delta=f"{acc_diff:+.1f}% vs Baseline",
            )
        with c2:
            unsupp_diff = (t_met["unsupported_rate_attempted"] - b_met["unsupported_rate_attempted"]) * 100
            st.metric(
                "Hallucination / Unsupported Rate",
                f"{t_met['unsupported_rate_attempted']*100:.1f}%",
                delta=f"{unsupp_diff:+.1f}% vs Baseline",
                delta_color="inverse",
            )
        with c3:
            st.metric(
                "Answerable Coverage",
                f"{t_met['answerable_coverage']*100:.1f}%",
                delta=f"{(t_met['answerable_coverage'] - b_met['answerable_coverage'])*100:+.1f}%",
            )
        with c4:
            st.metric(
                "Correct Abstention Rate",
                f"{t_met['correct_abstention_rate']*100:.1f}%",
                delta=f"{(t_met['correct_abstention_rate'] - b_met['correct_abstention_rate'])*100:+.1f}%",
            )

        st.markdown("### 📋 Formal Metric Comparison Table")
        comparison_data = {
            "Metric": [
                "Answer Accuracy",
                "Hallucination Rate (Attempted)",
                "Hallucination Rate (Total)",
                "Abstention Rate",
                "Answerable Coverage",
                "Correct Abstention Rate",
                "Abstention Precision",
                "Expected Calibration Error (ECE)",
                "Average Latency (s)",
            ],
            "Baseline RAG": [
                f"{b_met['accuracy']*100:.1f}%",
                f"{b_met['unsupported_rate_attempted']*100:.1f}%",
                f"{b_met['unsupported_rate_total']*100:.1f}%",
                f"{b_met['abstention_rate']*100:.1f}%",
                f"{b_met['answerable_coverage']*100:.1f}%",
                f"{b_met['correct_abstention_rate']*100:.1f}%",
                f"{b_met['abstention_precision']*100:.1f}%",
                f"{b_met['ece']:.3f}",
                f"{b_met['avg_latency']}s",
            ],
            "TrustRAG (Proposed)": [
                f"{t_met['accuracy']*100:.1f}%",
                f"{t_met['unsupported_rate_attempted']*100:.1f}%",
                f"{t_met['unsupported_rate_total']*100:.1f}%",
                f"{t_met['abstention_rate']*100:.1f}%",
                f"{t_met['answerable_coverage']*100:.1f}%",
                f"{t_met['correct_abstention_rate']*100:.1f}%",
                f"{t_met['abstention_precision']*100:.1f}%",
                f"{t_met['ece']:.3f}",
                f"{t_met['avg_latency']}s",
            ],
            "Interpretation / Target": [
                "Higher is better (correct answers on answerable + correct abstentions)",
                "Lower is better (unsupported answers among attempted)",
                "Lower is better (total unsupported / total questions)",
                "Context-dependent (should match unanswerable ratio)",
                "Higher is better (usable answers for answerable questions)",
                "Higher is better (abstention on unanswerable/misleading)",
                "Higher is better (appropriate abstentions / total abstentions)",
                "Lower is better (distance to perfect confidence calibration)",
                "Trade-off (TrustRAG adds auditing/verification calls)",
            ],
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        st.markdown("### 📈 Research Visualizations")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_comp = create_metric_comparison_chart(b_met, t_met)
            st.plotly_chart(fig_comp, use_container_width=True)

        with chart_col2:
            fig_ece = create_reliability_diagram(t_met["calibration_bins"], t_met["ece"])
            st.plotly_chart(fig_ece, use_container_width=True)

        st.markdown("#### Risk–Coverage Analysis")
        fig_rc = create_risk_coverage_chart(t_met["risk_coverage"])
        st.plotly_chart(fig_rc, use_container_width=True)

        st.caption(
            "Note on Calibration: ECE is computed across 5 bins. On smaller benchmark sample sizes (e.g. N < 50), "
            "ECE values should be interpreted as preliminary empirical observations rather than definitive bounds."
        )

        st.markdown("### 🔍 Sample-by-Sample Audit & Export")
        b_res = res["baseline_results"]
        t_res = res["trustrag_results"]

        export_rows = []
        for i in range(len(b_res)):
            export_rows.append({
                "Question": b_res[i]["query"],
                "Type": b_res[i]["question_type"],
                "Expected Answer": b_res[i]["expected_answer"],
                "Baseline Answer": b_res[i]["answer"],
                "Baseline Correct": b_res[i]["graded_correct"],
                "TrustRAG Answer": t_res[i]["answer"],
                "TrustRAG Abstained": t_res[i]["abstained"],
                "TrustRAG Correct": t_res[i]["graded_correct"],
                "TrustRAG Confidence": t_res[i].get("evaluated_confidence"),
                "TrustRAG Reason": t_res[i].get("abstention_reason"),
            })

        df_export = pd.DataFrame(export_rows)
        st.dataframe(df_export, use_container_width=True)

        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Detailed Evaluation CSV",
            data=csv_bytes,
            file_name="trustrag_evaluation_results.csv",
            mime="text/csv",
        )