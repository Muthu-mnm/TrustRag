# 🛡️ TrustRAG: Hallucination Detection & Abstention Calibration

> **Research-Oriented Evidence-Grounded Question-Answering System**  
> *Hallucination Detection and Abstention Calibration in Large Language Models Using Evidence-Grounded Question Answering.*  
> Powered by **Streamlit, LangChain, Local Ollama (`gemma3:4b`, `nomic-embed-text`), FAISS, and Plotly**.

---

## 📌 1. Project Overview & Research Context

Conventional Retrieval-Augmented Generation (RAG) pipelines force a Large Language Model to answer even when document evidence is missing, partial, or when the user poses an unanswerable or misleading query. This leads to **hallucinations, false confidence, and unsupported extrapolation**.

**TrustRAG** converts standard RAG into a calibrated, evidence-grounded research assistant:
1. **Page-Aware PDF Ingestion:** Uses `PyPDFLoader` to extract text while strictly preserving page numbers, chunk IDs, and document metadata.
2. **Pre-Generation Evidence Sufficiency Gate:** Prompts local **Gemma 3 4B** to audit whether retrieved passages contain verifiable facts to answer all material parts of the query *before* generating an answer.
3. **Composite Confidence Calibration:** Combines self-reported evidence sufficiency and normalized retrieval similarity into a transparent, documented score:
   $$\text{Confidence}(q, D) = (w_{\text{suff}} \times S_{\text{suff}}) + (w_{\text{ret}} \times R_{\text{ret}})$$
4. **Calibrated Abstention Policy:** Rejects out-of-scope, unanswerable, or misleading questions with an explicit refusal message and transparent reasoning:
   > *"I cannot provide a reliable answer because the available document evidence is insufficient."*
5. **Grounded Generation & Source Citations:** Enforces strict grounding with explicit document page citations (e.g. `[Page 2]`).
6. **Post-Generation Answer Verification:** An independent verification step inspects candidate claims against evidence, enforcing bounded retries or abstention upon hallucination.
7. **Empirical Evaluation Dashboard:** Benchmarks Baseline RAG vs. TrustRAG on Answerable, Unanswerable, and Misleading questions, measuring Accuracy, Hallucination Rate, Abstention Precision, Expected Calibration Error (ECE), and Risk–Coverage curves.

---

## 🛠️ 2. Technology Stack

- **Frontend:** Streamlit (ChatGPT-style dark UI with sidebar conversation history)
- **Programming Language:** Python 3.10+ / 3.12
- **Local LLM Runtime:** Ollama (100% offline, no cloud APIs or external tunnels)
- **Generation & Auditor Model:** `gemma3:4b` (compatible with `gemma4:26b` when available)
- **Embedding Model:** `nomic-embed-text` (768-dimensional local embeddings)
- **PDF Extraction:** `pypdf` (`PyPDFLoader` with page-aware chunking)
- **Vector Index:** `FAISS` (`faiss-cpu`) with normalized distance-to-similarity scoring
- **Evaluation & Charts:** `NumPy`, `Pandas`, `scikit-learn`, `Plotly`

---

## 🏛️ 3. System Architecture Comparison

### Baseline RAG Pipeline
```text
User PDF ──► PyPDFLoader ──► Chunks ──► nomic-embed-text ──► Vector Retrieval (Top-k)
                                                                     │
                                                                     ▼
                                                         Unconstrained Generation
                                                                     │
                                                                     ▼
                                                          Candidate Output
                                                     (High Hallucination Risk)
```

### TrustRAG Pipeline (Proposed)
```text
User PDF ──► Page-Aware Splitter ──► nomic-embed-text ──► FAISS Top-k Retrieval
                                                                  │
                                                                  ▼
                                                      [Evidence Sufficiency Gate]
                                                      (Auditor LLM JSON Output)
                                                                  │
                                          ┌───────────────────────┴───────────────────────┐
                                          ▼                                               ▼
                                  [Insufficient]                                     [Sufficient]
                                          │                                               │
                                          ▼                                               ▼
                                    ABSTAIN ◄───────── [Confidence Check (C < τ)] ────────┘
                                                                  │ (C ≥ τ)
                                                                  ▼
                                                      Strict Grounded Answer
                                                      (Includes Page Citations)
                                                                  │
                                                                  ▼
                                                      [Answer-Support Verifier]
                                                                  │
                                          ┌───────────────────────┴───────────────────────┐
                                          ▼                                               ▼
                                    [Unsupported]                                    [Supported]
                                          │                                               │
                                          ▼                                               ▼
                                 Bounded Retry / ABSTAIN                            Verified Answer
```

---

## 🧪 4. Live Experimental Validation on `Inception net.pdf`

The pipeline was validated against the paper *"Going deeper with convolutions"* (Szegedy et al., `Inception net.pdf`):

| Test Scenario | Query | Pipeline Behavior | Measured Outcome |
| :--- | :--- | :--- | :--- |
| **Document Indexing** | Ingest `Inception net.pdf` (1.2 MB) | Page-aware chunking & FAISS index | **12 pages, 67 chunks** indexed |
| **Answerable Question** | *"How many layers deep is the GoogLeNet architecture?"* | Sufficiency Gate **Passed** (Confidence: `0.83`) | **Grounded Answer with Citation:**<br>`"The GoogLeNet architecture is a 22-layer deep network [Page 2]."` |
| **Unanswerable Question** | *"What was the exact operating GPU temperature during GoogLeNet training?"* | Sufficiency Gate **Failed** (Confidence: `0.39` < `0.60`) | **Calibrated Abstention Triggered:**<br>`"I cannot provide a reliable answer because the available document evidence is insufficient."`<br>**Reason:** Missing GPU temperature data. |

---

## 📐 5. Mathematical Formulations & Metrics

### Composite Confidence Score
$$\text{Confidence}(q, D) = w_{\text{suff}} \cdot S_{\text{suff}} + w_{\text{ret}} \cdot R_{\text{ret}}$$

- $S_{\text{suff}} \in [0, 1]$: Self-reported confidence from LLM Evidence Auditor prompt.
- $R_{\text{ret}} \in [0, 1]$: Mean normalized retrieval similarity:
  $$R_{\text{ret}} = \frac{1}{K} \sum_{i=1}^K \frac{1}{1 + d_i}$$
- **Weights:** $w_{\text{suff}} = 0.65, w_{\text{ret}} = 0.35$ (summing to $1.0$).
- **Abstention Criterion:** The system abstains if $S_{\text{suff}} = \text{False}$, OR $\text{Confidence} < \tau$ (default $\tau = 0.60$), OR post-generation verification fails.

### Core Benchmark Metrics
1. **Answer Accuracy:**
   $$\text{Accuracy} = \frac{\text{Correct Answers on Answerable} + \text{Correct Abstentions on Unanswerable/Misleading}}{N}$$
2. **Hallucination / Unsupported Rate (Attempted):**
   $$\text{Unsupported Rate} = \frac{\text{Unsupported Attempted Answers}}{\text{Total Attempted Answers}}$$
3. **Answerable Coverage:**
   $$\text{Coverage} = \frac{\text{Attempted on Answerable}}{\text{Total Answerable Questions}}$$
4. **Correct Abstention Rate (Specificity):**
   $$\text{Correct Abstention} = \frac{\text{Abstentions on Unanswerable}}{\text{Total Unanswerable Questions}}$$
5. **Abstention Precision:**
   $$\text{Abstention Precision} = \frac{\text{Appropriate Abstentions}}{\text{Total Abstentions}}$$
6. **Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
7. **Risk–Coverage Frontier:** Parametric sweep evaluating coverage vs. empirical error rate as threshold $\tau$ varies.

---

## 🚀 6. Setup & Execution Guide (Windows)

### Step 1: Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Start Ollama & Pull Local Models
Verify that local Ollama is active on `http://localhost:11434`:
```powershell
# Pull local generation model
ollama pull gemma3:4b

# Pull local embedding model
ollama pull nomic-embed-text

# Verify installed models
ollama list
```

### Step 3: (Optional) Run Unit Tests
```powershell
python -m tests.test_trustrag
```

### Step 4: Launch the Streamlit Application
```powershell
streamlit run app.py
```
The application will open automatically in your browser at **`http://localhost:8501`**.

---

## 🖥️ 7. User Interface Overview

The interface is built with a ChatGPT-style layout:

### 1. 💬 Document Chat (Primary Workspace)
- **Sidebar Chat History:** Features a **`➕ New Chat`** button, list of clickable past queries, a **`🗑️ Clear History`** button, and active model status.
- **Document Ingestion:** Compact PDF uploader or one-click **`📄 Load 'Inception net.pdf' (Workspace Sample)`** button.
- **Grounded Responses:** Answers include document page citations (e.g. `[Page 2]`).
- **Abstention Callout:** Highlights refused queries with an explicit refusal box and auditor rationale.
- **Diagnostics Trace:** Expandable breakdown showing Evidence Sufficiency auditor output, confidence composition formula, verifier verdict, and retrieved passage snippets with similarity scores.

### 2. 📊 Evaluation Dashboard
- **Benchmark Loader:** Load custom CSVs or click **`📁 Load Built-in Benchmark (10 Qs)`** (covering Answerable, Unanswerable, and Misleading questions).
- **Metric Cards:** Accuracy, Hallucination/Unsupported Rate, Coverage, and Correct Abstention Rate.
- **Comparative Table:** Baseline RAG vs. TrustRAG side-by-side comparison with explicit denominators.
- **Plotly Visualizations:** Grouped metric bar chart, Reliability Diagram (ECE) with diagonal calibration reference, and Risk–Coverage curves.
- **Audit Export:** Download detailed per-sample evaluation results as a CSV.

---

## 📂 8. Repository Structure

```text
PDF-QA-RAG-System/
├── app.py                          # Streamlit Research Dashboard
├── requirements.txt                # Python package dependencies
├── README.md                       # Research paper-style documentation
├── Inception net.pdf               # Local benchmark evaluation PDF
├── data/
│   └── sample_evaluation.csv       # Curated 10-sample benchmark dataset
├── tests/
│   └── test_trustrag.py            # Unit test suite for confidence, JSON parsing & metrics
└── trustrag/                       # Core Modular TrustRAG Package
    ├── __init__.py
    ├── config.py                   # Runtime configurations, constants & prompt templates
    ├── ollama_client.py            # Robust Ollama HTTP API client
    ├── pdf_processor.py            # Page-aware PDF extraction (PyPDFLoader) & metadata chunking
    ├── vector_store.py             # FAISS indexing with nomic-embed-text & similarity scoring
    ├── evidence_gate.py            # Evidence sufficiency auditor LLM gate (JSON output)
    ├── confidence.py               # Composite confidence scoring & abstention policy manager
    ├── generator.py                # RAGPipeline (Baseline RAG vs TrustRAG with bounded retries)
    ├── verifier.py                 # Post-generation factual claim verifier
    ├── evaluator.py                # Benchmark evaluator (Accuracy, Unsupported rate, ECE, Risk-Coverage)
    └── visualization.py            # Plotly charts (Metric comparison, Reliability diagram, Risk-Coverage)
```

---

## ⚠️ 9. Research Integrity & Known Limitations

1. **LLM Self-Reported Confidence:** Confidence emitted by language models reflects token-level self-assessments, not calibrated Bayesian posteriors. It serves as an empirical gating signal.
2. **Retrieval Bottlenecks:** If the embedding model fails to retrieve a passage that exists in the PDF, TrustRAG will correctly identify lack of evidence and abstain. This represents a retrieval limitation rather than an inference error.
3. **Small-Sample ECE Sensitivity:** Expected Calibration Error (ECE) is sensitive to the number of bins ($M$) and dataset size ($N$). On small datasets ($N < 50$), ECE is an indicative signal rather than an absolute bound.
4. **Verification Errors:** The verifier is also an LLM and may occasionally misclassify complex reasoning. Critical production deployments require human spot-auditing.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
