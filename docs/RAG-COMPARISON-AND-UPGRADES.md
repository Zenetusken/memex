# Open-Source Local RAG & NotebookLM-Style Projects: Comparative Analysis & Upgrade Roadmap

This document analyzes comparable open-source Retrieval-Augmented Generation (RAG) and research-assistant frameworks, indexing them by their architectural and functional relevance to Memex. It details pipeline differences, evaluates their optimization tactics, and proposes a series of advanced theories and upgrades viable within a **12 GB VRAM** local hardware profile.

---

## 1. Competitive Index: Comparable Projects Ranked by Relevance

We evaluate comparable projects against Memex's core profile: **local-first, verified, high-fidelity multimodal ingestion, and extreme VRAM efficiency.**

| Project | GitHub Repo | Signature Feature | Memex Relevance | Architecture Alignment |
|---|---|---|---|---|
| **Kotaemon** | `Cinnamon/kotaemon` | Visual citations, multi-vector retrieval, dual-RAG pipelines. | **Extreme** (9.5/10) | Local-first, Gradio UI, note-centric, supports GraphRAG. |
| **RAGFlow** | `infiniflow/ragflow` | Layout-aware DeepDoc parsing, custom template segmentation. | **Very High** (8.5/10) | Deeply structured, PDF OCR, table structures. |
| **Khoj** | `khoj-ai/khoj` | Multi-source offline index (Obsidian, Github, Markdown), local sync. | **High** (8.0/10) | Conversational agents, notes-as-second-brain focus. |
| **NotebookLlama** | `run-llama/notebookllama` | PDF-to-Podcast conversational audio generation pipeline. | **Medium-High** (7.5/10) | LlamaIndex-backed pipeline, TTS script orchestration. |
| **SurfSense** | `Decentralised-AI/SurfSense` | Enterprise & SaaS local indexing connectors (Slack, Linear, GitHub). | **Medium** (6.0/10) | Highly customized search agent over cloud endpoints. |

---

## 2. In-Depth Comparative Pipeline Breakdown

### A. Kotaemon (Cinnamon)
- **Ingestion Tactics**: Uses standard loaders (PyMuPDF, Unstructured) but incorporates **multi-vector/parent-child indexing**. It creates small child chunks (e.g., 100–200 tokens) for highly semantic vector mapping, but links them to their original parent sections (e.g., 1000–2000 tokens) which are passed as context to the LLM.
- **Answering & Verification**: Standard RAG with cross-encoder reranking. It supports a "visual citation" feature where clicking a citation opens a PDF page viewer and draws a highlighted bounding box directly on the text segment.
- **VRAM/Model Management**: Relies heavily on a running Ollama instance. It does not actively manage co-resident VRAM state, which frequently leads to CUDA OOMs on 12 GB rigs if both embedding, reranking, and generation run concurrently.

### B. RAGFlow (InfiniFlow)
- **Ingestion Tactics**: Relies on specialized vision-layout models (YOLOv8-based layout detection) to segment documents before OCR. Rather than using fixed token chunking, it segments by document layout templates (e.g., *Book, Presentation, Table, Q&A*). If a table is detected, it preserves column/row coordinate mappings and performs auto-rotation for scanned tables based on OCR confidence.
- **Answering & Verification**: Focuses heavily on high-precision keyword/vector fusion. It does not enforce a strict counterfactual refusal gate; it prioritizes dense layout preservation to prevent the model from misreading tabular structures.
- **VRAM/Model Management**: Intended as an enterprise platform, usually deployed via a heavy Docker Compose stack. Running its full suite of layout models, OCR, and generator models concurrently requires substantial hardware, making it poorly optimized for single-user 12 GB consumer GPUs.

### C. NotebookLlama (Meta/Together.ai)
- **Ingestion Tactics**: PDF to plain Markdown extraction.
- **Answering & Audio Generation**: Pipeline consists of three sequential LLM prompts: (1) Ingest the markdown and write a high-level summary. (2) Convert the summary into a highly conversational, engaging podcast script featuring a Host (skeptical, inquisitive) and a Co-Host (expert explainer). (3) Format the script with SSML/TTS cues.
- **Audio Synthesis**: Feeds the script line-by-line to a text-to-speech model (e.g., Bark, XTTS, or Kokoro) to generate distinct WAV files, then merges them with background transition music.

---

## 3. Advanced Theories & Standalone Upgrades for Memex

The following upgrades represent highly viable research and development theories that fit within Memex's **12 GB VRAM constraint** by leveraging time-slicing and memory-efficient CPU fallback models.

### Upgrade Theory A: Parent-Child (Multi-Vector) Retrieval

> **AUDIT VERDICT (2026-06-10): NO-GO as written** — see [`docs/audits/16-parent-child-retrieval-audit.md`](audits/16-parent-child-retrieval-audit.md). The text-swap variant breaks the verify gate's per-chunk grounding contract; 150-word children churn every chunk_id (full-vault migration) against measured-saturated retrieval recall; an 800-word parent is truncated to 37% by the answer prompt's 1800-char budget. The reshaped **additive neighbor-window augmentation** variant is under measurement via `scripts/parent_context_probe.py` (GO gate: >10% headroom on any eval corpus).
* **Concept**: When indexing, generate two sets of chunks: 
  1. **Semantic Chunks** (150 words): Embedded in LanceDB. High vector-space density ensures exact, targeted retrieval.
  2. **Parent Chunks** (800 words, containing the child): Stored in the SQLite database.
* **Why it works on a 12 GB Rig**: High-density 150-word chunks are extremely cheap to retrieve and run through the CPU-resident or GPU-resident reranker. Only the final top-$k$ candidates (e.g., $k=5$) have their parent windows expanded from SQLite. This provides the orchestrator with wide, high-context boundaries (crucial for Qwen3.5-4B's 8,192 token window) without bloating vector database memory or slowing down the reranker.
* **Implementation Hook**: Adapt `@/home/drei/project/Doc_Flo/src/memex/index/pipeline.py` to write both `chunk` and `parent_chunk_id` fields, and `@/home/drei/project/Doc_Flo/src/memex/retrieve/hybrid.py` to retrieve the parent context from SQLite right before generation.

### Upgrade Theory B: RAPTOR (Recursive Tree-Organized Summarization)
* **Concept**: To solve Mei's literature review synthesis needs, we can construct recursive document trees. Chunks are clustered semantically (e.g., via Gaussian Mixture Models on EmbeddingGemma vectors). We call the 4B local orchestrator to write a dense summary of each cluster. Those summaries are clustered and summarized again, building a hierarchical tree.
* **Why it works on a 12 GB Rig**: Constructing the tree is an ingestion-time task. Since Memex uses an **exclusive-GPU ingestion mode** (temporarily pausing vLLM/answering to free the entire VRAM budget), we have the full 12 GB free to run Qwen3.5-4B for clustering and abstractive summarization. During retrieval, we search both leaf-level chunks and summary nodes, allowing the model to answer high-level thematic questions (e.g., "What are the common methodological disagreements in my vault?") without overwhelming the context window.

### Upgrade Theory C: Local "Deep Dive" Podcast Generation (NotebookLM-Style)
* **Concept**: Replicate the iconic NotebookLM audio generation. Add a CLI command `memex podcast --doc <id>` or a button on the Web UI document page.
* **Why it works on a 12 GB Rig**:
  1. **Script Generation**: Qwen3.5-4B is highly capable of structured dialogue generation.
  2. **Audio Synthesis**: Use **Kokoro-82M** as the TTS engine. Kokoro-82M is a state-of-the-art open-source TTS model with only 82 million parameters. It runs comfortably on a single CPU core or under 150 MB of VRAM, delivering hyper-realistic, expressive human-like speech. Because the TTS generation can run sequentially (sentence-by-sentence) after the script is written, it never needs to co-reside with a heavy generator model.

### Upgrade Theory D: Visual Citations with Bounding Box Highlights
* **Concept**: On the Web UI's side-by-side preview panel, clicking a grounded claim's citation doesn't just scroll to the PDF page—it renders a highlighted bounding box directly over the matching text segment on the page image.
* **Why it works on a 12 GB Rig**: This is a pure CPU/Frontend feature with **0 VRAM overhead**. Since Memex already tracks true character starts and page boundaries to align slide decks and transcripts (ADR-0018), and generates server-rendered page images via PyMuPDF/pdfium, we can extract bounding-box coordinates during parse and store them in the per-doc manifest. The frontend uses a simple CSS SVG overlay over the page JPEG.

### Upgrade Theory E: Multi-Template Layout-Aware Segmenter (RAGFlow-Style)
* **Concept**: Introduce preset parsing profiles at ingestion time (e.g., `--profile slide-deck`, `--profile source-code`, `--profile technical-manual`). 
* **Why it works on a 12 GB Rig**: Memex already contains custom symbol boundary chunking for code and ASR routes for audio. Formalizing this into a multi-template router allows us to tailor chunk overlap and table parsing constraints strictly via CPU-bound heuristic parsers (Docling/PyMuPDF rules) without needing heavy, always-on vision models.

---

## 4. Prioritized Research Backlog

The following table lists the recommended research spikes to validate these theories, ranked by implementation cost and user value.

| Upgrade | Expected Benefit | VRAM Impact | Est. Code Footprint | Target Evaluation |
|---|---|---|---|---|
| **A. Visual Citations** | Visual confirmation of grounding (highly engaging UX). | 0 MB | Small (Web UI CSS overlays, manifest coordinates) | Slide-deck PDF page previews. |
| **B. Parent-Child Retrievals** | Wider context window without vector dilution. | 0 MB | Medium (Database migrations, retrieval updates) | CCNA slide decks & 10-K financial tables. |
| **C. Kokoro-82M Podcast** | Offline "Deep Dive" audio summaries (the true NotebookLM signature). | <150 MB (sequential execution) | Medium (Script prompt, TTS python runner, audio merge) | Selected multi-doc research scopes. |
| **D. RAPTOR Trees** | Comprehensive cross-document synthesis. | 0 MB at retrieval, sequential LLM load at ingest. | High (Semantic clustering, recursive indexing, multi-node retrieval) | Mei's literature review (400-paper synthesis). |
