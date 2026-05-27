# Memex

### A local-first, fully agentic system for turning documents into knowledge that belongs to you

---

## As We May Think, Again

In 1945, Vannevar Bush described a machine he called the **Memex** — a desk-sized device that would let a researcher store every book, paper, and note they had ever encountered, and then trace associative trails between them at the speed of thought. The web borrowed Bush's idea of the hyperlink but quietly dropped his original premise: the Memex was personal. It lived on your desk. It belonged to you. No one mined your trails for advertising. No one held your reading hostage behind a subscription.

Eighty years later, we still don't have what Bush described. We have things that look like it from a distance — search engines, cloud notes apps, AI assistants — but every one of them trades the original promise for convenience. Your documents live on someone else's servers. Your reading history trains someone else's models. Your "second brain" stops working the day a startup pivots or a billing cycle lapses.

Memex is an attempt to finally build the thing Bush described, using the tools we now have: open-weight language models small enough to run on a single consumer GPU, vector retrieval that fits on a laptop, and agentic orchestration that turns a folder of PDFs into a navigable knowledge graph — all without a single byte leaving your machine.

---

## The Problem

We have more information than at any point in human history, and most of it is stuck. PDFs that can't be searched. Scans that can't be linked. Lecture slides that exist only as exported screenshots. Handwritten lab notebooks that never get transcribed. Internal documentation scattered across a dozen formats. Prompt libraries copy-pasted across Discord, Notion, and Obsidian, with no connection between them.

The existing options for unsticking that information all extract a price the original document was never willing to pay:

- **Cloud OCR services** require uploading documents that may be confidential, regulated, or simply not yours to share.
- **Frontier AI assistants** offer to "chat with your PDFs" — and in doing so, ingest them into someone else's context window, training pipeline, or retention policy.
- **Enterprise knowledge platforms** lock you into proprietary block formats, per-seat pricing, and APIs that can be rate-limited or deprecated at will.
- **Consumer notes apps** treat documents as second-class citizens — fine for snippets, hostile to the structure of a real research corpus.

Consider Mei, a PhD student writing a literature review across four hundred papers. She needs to find every mention of a specific assay across the corpus, see which papers cite which, and surface the methodological disagreements between them. Her current options are: read everything manually (the historical default), pay for an enterprise tool her grant won't cover, or hand the entire corpus to a cloud LLM and hope the embargoed preprints don't end up in a training set.

Or consider Daniel, a staff engineer maintaining documentation-as-code for an internal platform. He wants his team's docs, ADRs, RFCs, and prompt library to be queryable as a single corpus — but he can't ship them to a third-party service, and the off-the-shelf RAG tools assume he's willing to.

Memex is the option neither Mei nor Daniel currently has: professional-grade document understanding, semantic search, and agentic question-answering that runs entirely on local hardware, outputs to plain Markdown, and never makes a network call it didn't have to.

---

## Why Markdown, Not Notion (Not Anything Else)

The first version of this project ended in Notion. That was a mistake — not a small one, a foundational one. A system whose entire reason for existing is local-first privacy cannot make its terminal output a cloud-hosted, API-rate-limited, proprietary block store. The contradiction had to go.

The replacement is the most boring and most powerful format we have: **plain Markdown, on a local filesystem.**

Markdown isn't a compromise. It's a strictly better target for this problem:

- **It is the native language of LLMs.** Modern models read and write Markdown more reliably than any other structured format. Round-tripping document → Markdown → agent → Markdown is lossless and fast.
- **It composes with everything.** Obsidian, Logseq, VSCode, Zed, Cursor, ripgrep, git, static-site generators, every RAG framework ever written, and every future tool worth using. Notion composes with Notion.
- **Frontmatter gives you taxonomy for free.** YAML at the top of each file handles tags, dates, sources, authors, custom schemas, citation keys — no "database property" ceremony required.
- **Wikilinks give you a graph for free.** `[[entity]]` syntax produces a knowledge graph as a side effect of just writing notes. The graph is implicit in the corpus; no separate database has to be the source of truth.
- **It survives.** A folder of `.md` files will be readable in 2050. A Notion export from 2026 probably won't be.
- **It is the lingua franca of docs-as-code.** Memex's output drops directly into any documentation pipeline, prompt library repo, or research vault that already exists.

A processed Memex corpus looks like this on disk:

```
vault/
  documents/
    2024-smith-drug-interactions.md      # frontmatter + content + [[wikilinks]]
    2024-smith-drug-interactions/        # figures, tables, attachments, original PDF
  prompts/
    extraction-prompts/
  .memex/
    embeddings.lance                     # LanceDB vector index
    graph.kuzu                           # entity & citation graph
    search.sqlite                        # SQLite FTS5 full-text index
    traces/                              # Langfuse-compatible run traces
    manifest.json                        # processing provenance per file
```

The `documents/` tree is the **source of truth**. The `.memex/` sidecar is **derived state** — it can be deleted and rebuilt from the Markdown at any time, on any machine, with no loss. This is the inversion of every cloud knowledge product: the data is yours, the indexes are disposable.

---

## Our Vision

We are building toward four convictions.

**Privacy by construction, not by policy.** Most "private" software is private because someone promises it is. Memex is private because there is no remote endpoint to leak to. Air-gap the machine and the product still works exactly the same. Privacy that depends on a privacy policy is not privacy.

**Small models, used well, beat big models used carelessly.** A 7-billion-parameter model running locally, given a tight scope and good tools, will outperform a frontier model called five times against a poorly designed pipeline. Memex is a bet that the next decade of useful AI is in disciplined orchestration of open-weight models, not in renting larger ones.

**Documents are structured communications, not strings of text.** A table is structured data. A footnote is a linked annotation. A figure caption belongs to a figure. An equation is mathematics. Memex preserves this structure end-to-end — extraction, storage, retrieval, and answers all respect that documents have shape.

**Open source as a complete commitment.** Not open-core. Not source-available. Not "open weights, closed orchestration." The engine, the agents, the indexes, the schemas, the prompts — everything that processes your documents is inspectable, modifiable, and forkable. If we stop maintaining it, you don't lose anything.

---

## Core Principles

### 1. Local-First, By Construction

Processing happens on your hardware. There is no fallback to the cloud, no "premium" tier that uses a frontier model, no telemetry that phones home with summaries of what you've been reading. The reference deployment runs disconnected. This isn't a feature toggle — it's the architecture.

### 2. Markdown as the Source of Truth

The processed Markdown corpus is the authoritative artifact. Embeddings, graphs, and search indexes are derived state and are explicitly disposable. Any operation Memex can do can also be done by another tool reading the same Markdown — Memex is one possible interface to your vault, not its prison.

### 3. Small Models, Used Well

Memex commits to running on a single consumer GPU (the reference target is a 12GB card). This forces good engineering: tight prompts, structured outputs, retrieval that earns its tokens, agent loops with budgets, and verification steps that catch model error before it propagates. The constraint produces the discipline.

### 4. Observable at Every Layer

Every parsing decision, every retrieval, every agent step, every model call, every token is traced, timestamped, and inspectable. Open the trace viewer and you can replay exactly what happened when Memex answered a question — which chunks it pulled, which it rejected, which model produced which output, how long each step took. AI you can't audit is AI you can't trust.

### 5. Composable, Not Captive

Memex exposes its corpus through an MCP server. Any MCP-compatible agent — Claude Code, Cursor, your own — can query the vault. The vault itself is just Markdown, so any tool that reads files reads Memex. There is no proprietary surface area we depend on you depending on.

---

## The Stack

Memex is opinionated about its stack because constraint at this level is what makes the system viable on consumer hardware. There is no model picker. There is one default configuration that we will keep optimizing as the open-weight landscape evolves. Everything here is open weights, open source, and runs offline.

**Document understanding**

- **Docling** (IBM, Apache 2.0) as the primary parsing pipeline. Handles layout, tables, equations, and outputs structured Markdown directly. CPU-first with optional GPU acceleration.
- **Qwen3-VL-8B-AWQ** (~7.4 GB, served via a short-lived parse-time vLLM process) as the vision-language fallback for pages Docling can't handle confidently — scanned handwriting, dense diagrams, directed flow/state diagrams, unusual layouts.

**Reasoning and agents**

- **Qwen3-8B-Instruct** (Q4_K_M, ~5GB VRAM) as the orchestrator and answer model. Strong tool-use, structured-output reliability, multilingual.
- **LangGraph** for state-machine orchestration. Agent loops are explicit graphs with budgets, not free-form ReAct chains.

**Retrieval**

- **EmbeddingGemma 300M** for dense embeddings. Small, multilingual, fast enough to embed a corpus of thousands of pages in minutes on the 4070.
- **bge-reranker-v2-m3** for second-stage reranking on candidates returned by hybrid search.
- **SQLite FTS5** for keyword search; **LanceDB** for vectors; combined via reciprocal rank fusion.

**Knowledge graph**

- **RyuGraph** (embedded, MIT — the maintained fork of Kuzu after the upstream archival; see ADR-0005) for the entity and citation graph. Cypher queries, columnar storage, no server to run.

**Inference**

- **vLLM** as the inference server (OpenAI-compatible API, paged attention, good throughput on Ada-generation GPUs). For lighter use, **Ollama** as a drop-in alternative.

**Observability**

- **Langfuse** (self-hosted, MIT) for tracing every agent run, model call, retrieval, and tool invocation. Every answer Memex gives is reproducible from its trace.

**Interop**

- An **MCP server** exposing the vault as queryable tools (search, retrieve, follow links, summarize, cite). Any MCP client speaks to Memex.

VRAM budget on the reference RTX 4070 (12GB), with the agent and embedding models co-resident: orchestrator ~5GB, embedder ~600MB, reranker ~600MB, KV cache and overhead ~3GB. The VLM fallback (Qwen3-VL-8B-AWQ, ~7.4GB) runs only at *parse* time, in its own short-lived vLLM process on the GPU freed by pausing the orchestrator — never co-resident with answering. Inference is sequential by design — the agent is the bottleneck, not parallel decoding.

---

## How Memex Works

Memex is not "OCR plus a chatbot." It is an agent loop with bounded scope and explicit verification.

When you drop a document into the vault, the parsing agent first inspects layout and content type — text-heavy, scan-heavy, equation-heavy, slide deck, handwritten — and routes accordingly. Docling handles the long tail of clean PDFs and DOCX files. Hard pages are escalated to the VLM. Tables are extracted as structured Markdown tables, not as flattened text. Equations become LaTeX. Figures are extracted, captioned by the VLM, and stored alongside the document.

The resulting Markdown is then enriched: entities are extracted, citations are resolved against the rest of the vault, wikilinks are inserted where the agent has high confidence, and the document's frontmatter is populated with metadata. Every enrichment is recorded in the manifest with its source — model, prompt, confidence — so you can audit or revert any decision later.

When you query the vault, the answering agent retrieves hybrid candidates (BM25 plus vector), reranks them, and constructs an answer with inline citations pointing back to the exact source span. If the agent isn't confident it can answer, it says so — and shows you what it did find. There is no hallucinated confidence. There is no answer without a citation trail.

Every step of every loop emits a trace. Open Langfuse and you can see the full reasoning, the tool calls, the retrieved chunks, the rejected candidates, and the tokens spent. This is the difference between using AI and trusting AI.

---

## What Makes Memex Different

**No cloud, no exceptions.** Every other "private RAG" tool has an asterisk somewhere — a managed inference endpoint, a hosted embedding API, telemetry "for product improvement." Memex doesn't. The asterisk is the whole product.

**Markdown out, not lock-in out.** Memex produces a vault that is useful even if you stop using Memex tomorrow. The Markdown is yours. The graph is reproducible. The embeddings are regenerable. There is no migration story because there is nothing proprietary to migrate from.

**Tuned for one rig.** Memex picks a hardware target (12GB consumer GPU) and tunes for it ruthlessly. Most local-AI projects either assume an H100 or run so slowly on consumer hardware that they're toys. Memex is built to be usable, every day, on the machine you already own.

**Agentic from the ground up.** Parsing, enrichment, and querying are all agent loops — bounded, verified, observable, but genuinely agentic. The system can re-attempt, escalate, or refuse. It is not a one-shot pipeline pretending to be intelligent.

**Honestly open source.** Apache or MIT throughout, no contributor license agreement designed to enable a future re-license, no proprietary "pro" tier. If we lose interest, the project is still useful. That is the only meaningful test of an open-source commitment.

---

## Who Memex Is For

**Researchers and graduate students** managing literature reviews across hundreds or thousands of papers, who need semantic search, citation graphs, and cross-paper question-answering without uploading embargoed or sensitive work to a third party.

**Students** building durable study corpora from lecture notes, slides, textbooks, and their own annotations — a personal knowledge base that accumulates across a degree rather than evaporating at the end of each semester.

**Documentation-as-code teams** maintaining technical docs, ADRs, RFCs, runbooks, and internal references as Markdown in git. Memex layers semantic search and agentic Q&A on top of an existing docs repo without changing the source format.

**Prompt library maintainers** organizing growing collections of prompts, evaluations, and outputs into something queryable. Memex treats a prompt library as a first-class document type — every prompt indexed, tagged, linked to its evaluations, and retrievable by intent rather than filename.

**Independent technical writers, archivists, and analysts** working with material they cannot ship to a cloud service — under embargo, under contract, under regulation, or just under principle.

The original document listed lawyers, doctors, and government workers. Those use cases are real, but they require certifications and audits beyond the scope of an open-source project. Memex serves them by being correct, observable, and local — not by claiming compliance it cannot underwrite.

---

## Success Metrics

We measure success in technical quality, in usability on real hardware, and in whether the project survives long enough to matter.

**Quality**

- Layout-faithful extraction on standard documents: 98%+ structural fidelity (tables-as-tables, equations-as-LaTeX, headings preserved).
- Citation precision on retrieval-grounded answers: 95%+ of cited spans support the cited claim.
- Hallucination rate on out-of-corpus questions: the agent declines to answer rather than fabricating, in 99%+ of cases.

**Performance on the reference rig (RTX 4070, 32GB RAM)**

- 100-page text-heavy PDF: end-to-end ingestion in under 4 minutes.
- 100-page scan-heavy PDF: end-to-end ingestion in under 12 minutes.
- Interactive query latency: first token in under 2 seconds, complete answer with citations in under 15 seconds for a typical question over a 500-document vault.

**Adoption and durability**

- A community of contributors large enough that Memex is no longer single-maintainer fragile.
- A plugin ecosystem for domain-specific extractors (citation styles, code-aware parsers, lab notebook formats).
- Survival: still actively useful and maintained three years from launch, which is the real test.

We do not measure success in monthly active users we can sell ads to. We do not have ads.

---

## The Roadmap

**Now**: end-to-end local pipeline — Docling parsing, Qwen-based agents, hybrid retrieval, Markdown vault, MCP server, Langfuse traces.

**Next**: smarter cross-document reasoning. The agent learns to traverse the citation graph during answering, not just retrieve flat chunks. Methodological disagreements between papers become first-class queryable structures.

**After that**: incremental re-indexing as the vault grows, so a thousand-paper corpus doesn't require a full rebuild when one PDF is added. Speculative parsing on idle GPU time. Domain plugins maintained by the people who actually work in those domains.

**Long-horizon**: documents that maintain their own state. A paper added today knows what cites it tomorrow. The vault becomes a living graph, not a static archive — closer, finally, to what Bush actually described.

---

## An Invitation

Memex is being built in the open, by people who want this thing to exist for themselves first. If you process documents for a living, study them for a degree, write them for a team, or just have a hard drive full of PDFs you've never been able to search — this is for you to use, fork, criticize, or contribute to.

The web turned Bush's Memex inside out: instead of a personal device with associative trails, we got a global commons whose trails are owned by advertisers. We are not going to undo that. But we can, with the open-weight models and consumer hardware we now have, build the thing that was supposed to come first — the private, personal, associative machine that turns your documents into something you can actually think with.

Welcome to Memex.

---

> _"Consider a future device for individual use, which is a sort of mechanized private file and library. A memex is a device in which an individual stores all his books, records, and communications, and which is mechanized so that it may be consulted with exceeding speed and flexibility. It is an enlarged intimate supplement to his memory."_
>
> — Vannevar Bush, *As We May Think*, 1945

> _"The best way to predict the future is to build it."_
>
> — Alan Kay

**Memex** — *Your documents. Your machine. Your trails.*
