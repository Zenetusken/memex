# Memex

### Self-hosted RAG orchestration on a single 12 GB GPU — turning documents, recordings, images, and code into knowledge that belongs to you, and answers you can trust

---

## As We May Think, Again

In 1945, Vannevar Bush described a machine he called the **Memex** — a desk-sized device that would let a researcher store every book, paper, and note they had ever encountered, and then trace associative trails between them at the speed of thought. The web borrowed Bush's idea of the hyperlink but quietly dropped his original premise: the Memex was personal. It lived on your desk. It belonged to you. No one mined your trails for advertising. No one held your reading hostage behind a subscription.

Eighty years later, we still don't have what Bush described. We have things that look like it from a distance — search engines, cloud notes apps, AI assistants — but every one of them trades the original promise for convenience. Your documents live on someone else's servers. Your reading history trains someone else's models. Your "second brain" stops working the day a startup pivots or a billing cycle lapses.

Memex is an attempt to finally build the thing Bush described, using the tools we now have: open-weight language models small enough to run on a single 12 GB consumer GPU, vector retrieval that fits on a laptop, and agentic orchestration that turns a folder of PDFs, lecture recordings, screenshots, and source code into a navigable, verified knowledge base — all without a single byte leaving your machine.

---

## The Problem

We have more information than at any point in human history, and most of it is stuck. PDFs that can't be searched. Scans that can't be linked. Lecture recordings nobody will ever re-watch. Slide decks that exist only as exported screenshots. Photographed whiteboards and pasted diagrams. Handwritten lab notebooks that never get transcribed. Codebases whose institutional knowledge lives in nobody's head. Internal documentation scattered across a dozen formats.

The existing options for unsticking that information all extract a price the original document was never willing to pay:

- **Cloud OCR services** require uploading documents that may be confidential, regulated, or simply not yours to share.
- **Frontier AI assistants** offer to "chat with your PDFs" — and in doing so, ingest them into someone else's context window, training pipeline, or retention policy.
- **Enterprise knowledge platforms** lock you into proprietary block formats, per-seat pricing, and APIs that can be rate-limited or deprecated at will.
- **Consumer notes apps** treat documents as second-class citizens — fine for snippets, hostile to the structure of a real research corpus.

And beneath the format problem sits a deeper one that none of those options even attempt to solve: **every "chat with your documents" tool will answer your question, but almost none will tell you when the answer isn't actually in your documents.** A retrieval pipeline bolted to a language model produces fluent, confident, citation-shaped text whether or not the sources support it. For anything that matters — a literature review, a compliance question, a number that goes into a report — an unverified answer is worse than no answer, because it carries false authority.

Consider Mei, a PhD student writing a literature review across four hundred papers. She needs to find every mention of a specific assay across the corpus, see which papers cite which, and surface the methodological disagreements between them — and when she quotes a finding, she needs to know the system didn't invent it. Her current options are: read everything manually (the historical default), pay for an enterprise tool her grant won't cover, or hand the entire corpus to a cloud LLM and hope the embargoed preprints don't end up in a training set.

Or consider Daniel, a staff engineer maintaining documentation-as-code for an internal platform. He wants his team's docs, ADRs, RFCs, prompt library, and the codebase itself to be queryable as a single corpus — "where is this payload serialized?" answered with the exact function, not a paraphrase — but he can't ship any of it to a third-party service, and the off-the-shelf RAG tools assume he's willing to.

Or consider Sofia, a student with a semester of recorded lectures and the slide decks that went with them. The knowledge is split across the two: the slide states the claim, the lecturer's spoken aside explains it. She needs both searchable together, with answers that cite the slide page and the minute mark.

Memex is the option none of them currently has: professional-grade understanding of every format they actually have — PDFs, scans, Office files, images, audio, video, source code — plus semantic search and agentic question-answering whose every claim is either grounded in a cited source or honestly refused. It runs entirely on local hardware, outputs to plain Markdown, and never makes a network call it didn't have to.

---

## Beyond RAG

It is tempting to file Memex under "local RAG," and wrong. Retrieval-augmented generation is a component of Memex the way a fuel pump is a component of a car — necessary, present, and not the point.

The point is **verification**. Memex treats "the model said so" as insufficient grounds for presenting anything as fact:

- **Refusal is a first-class outcome.** When the retrieved evidence doesn't ground an answer, the agent returns a structured refusal with its reasoning — not a plausible guess. This property is enforced by a counterfactual evaluation gate: questions whose answers are deliberately absent from the corpus must be refused, and that gate must hold before changes ship.
- **Every claim carries a citation that is checked, not decorated.** Answers are decomposed into claims, each bound to a source chunk, and a verification step rejects claims their cited chunk doesn't support — including a deterministic backstop that demotes any large figure absent from the cited table, because a single LLM check can be talked into rubber-stamping a fabricated total.
- **Numbers come from SQL, not from token prediction.** Aggregate and superlative questions over tables ("total fees paid to all directors") run a structured text-to-SQL pass over a per-vault table store built at index time. A computed aggregate is presented only when an independent recomputation over the original cells agrees; otherwise the system falls back to ordinary retrieval — which refuses if it can't ground.
- **Analysis is fenced and labelled.** For genuinely synthetic questions the vault can't ground, an opt-in surface reasons from model knowledge — then extracts the discrete claims that reasoning made and re-verifies each one, in isolation, against the vault. Verified claims are presented as cited; everything else stays inside a clearly labelled "ungrounded analysis" block. The boundary between knowledge and speculation is visible in the interface itself.
- **Over-refusal is also a bug.** A system that refuses when the answer is plainly present is as unreliable as one that fabricates. Memex measures and tunes both failure directions — the gate is calibrated, not merely conservative.

None of this is achievable with a retrieval pipeline alone. It requires an agent loop with explicit verification nodes, deterministic backstops where LLM judgment is known to fail, and an evaluation harness that treats honesty as a hard gate rather than a vibe. That loop — not the vector database — is what Memex is.

---

## The 12 GB Thesis

The other half of Memex's value is where all of that runs: **the entire system — parsing, vision, speech-to-text, embeddings, reranking, the knowledge graph, and the verified answering agent — self-hosted on a single 12 GB consumer GPU.** Not a degraded "lite" tier of a datacenter product, not a demo that assumes an H100. The reference deployment is an RTX 4070, the card people actually own, and the system is tuned for it ruthlessly.

That doesn't happen by picking small models and hoping. It happens by **orchestrating the GPU the way an operating system schedules a CPU** — and this orchestration layer is the engineering core of the project:

- **Time-slicing, not co-residence, for the heavy models.** The 8B vision-language model that transcribes diagrams and scanned pages could never fit beside the answering stack — so it doesn't try. At parse time the orchestrator is paused, its VRAM freed, and the VLM runs in its own short-lived process; when parsing ends, the orchestrator comes back. Ingestion is an exclusive-GPU mode: while a document is being consumed, every byte of VRAM goes to the pipeline, and answering resumes the moment it lands.
- **Named modes for the swing space.** On a 12 GB card the orchestrator's context window and the GPU-resident reranker compete for roughly 3 GB. Memex names that tradeoff as switchable co-residence modes — `fast` (GPU reranker, low latency) and `full` (CPU reranker, wide context) — and the default `auto` mode reads *live* free VRAM at load and adapts. Placement is correctness-neutral by construction: the reranker's ordering is byte-identical on CPU and GPU, so a mode changes latency, never the answer.
- **Every resident model earns its bytes.** The orchestrator is a 4-bit-quantized 4B model (~6.3 GB live); the embedder and reranker together take ~1.2 GB; quantization kernels and KV-cache budgets are chosen per model, measured, and re-validated against the full evaluation gates whenever they change. There is no model picker — there is one configuration that is known to fit and known to pass.
- **Graceful degradation instead of OOM.** A reranker that hits an out-of-memory retries at batch size one; a placement that can't fit falls back to CPU; an exhausted budget surfaces as a typed error, not a crashed answer. The system bends under pressure — it does not break mid-query.
- **Sequential inference by design.** The agent loop, not parallel decoding, is the bottleneck — so VRAM is spent on capability and context rather than concurrent throughput. One user, one rig, the best answer that rig can produce.

This is the claim behind "small models, used well": frontier-style document understanding and verified question-answering are achievable on commodity hardware — but only if the scarce resource is scheduled with the same discipline as the answers are verified. The 12 GB budget isn't a limitation Memex tolerates. It is the design constraint that produces the whole architecture — and the reason the product is genuinely *self-hosted* rather than nominally so.

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
    2024-smith-drug-interactions/        # figures, attachments, original PDF
    2026-lecture-12-routing.md           # timestamped transcript of a recorded lecture
    payments-handlers-rs.md              # source code, stored verbatim, chunked by symbol
  .memex/
    embeddings.lance                     # LanceDB vector index
    search.sqlite                        # SQLite FTS5 full-text index
    tables.sqlite                        # structured table store (text-to-SQL)
    graph.ryu                            # entity & citation graph
    manifests/                           # per-document processing provenance
```

The `documents/` tree is the **source of truth**. The `.memex/` sidecar is **derived state** — it can be deleted and rebuilt from the vault at any time, on any machine, with no loss (the indexes regenerate from the Markdown; the one class of derived block the content-only `.md` can't reconstruct alone — non-deterministic chart-OCR — is cached in the manifest sidecar and re-derivable by re-parsing the retained source, ADR-0003 #362). This is the inversion of every cloud knowledge product: the data is yours, the indexes are disposable.

---

## Our Vision

We are building toward five convictions.

**Privacy by construction, not by policy.** Most "private" software is private because someone promises it is. Memex is private because there is no remote endpoint to leak to. Air-gap the machine and the product still works exactly the same. Privacy that depends on a privacy policy is not privacy.

**Small models, used well, beat big models used carelessly.** A 7-billion-parameter model running locally, given a tight scope and good tools, will outperform a frontier model called five times against a poorly designed pipeline. Memex is a bet that the next decade of useful AI is in disciplined orchestration of open-weight models, not in renting larger ones.

**Documents are structured communications, not strings of text.** A table is structured data. A footnote is a linked annotation. A figure caption belongs to a figure. An equation is mathematics. A function is a symbol with a boundary. A lecture is speech anchored to a timeline and a slide. Memex preserves this structure end-to-end — extraction, storage, retrieval, and answers all respect that documents have shape.

**An answer you can't verify is not an answer.** Fluent text is cheap; grounded text is the product. Every claim Memex presents traces to a cited source span that survived a verification check, and when the evidence isn't there, the honest output is a refusal that shows what was found instead. Truth over fluency, every time.

**Open source as a complete commitment.** Not open-core. Not source-available. Not "open weights, closed orchestration." The engine, the agents, the indexes, the schemas, the prompts — everything that processes your documents is inspectable, modifiable, and forkable. If we stop maintaining it, you don't lose anything.

---

## Core Principles

### 1. Local-First, By Construction

Processing happens on your hardware. There is no fallback to the cloud, no "premium" tier that uses a frontier model, no telemetry that phones home with summaries of what you've been reading. The reference deployment runs disconnected. This isn't a feature toggle — it's the architecture.

### 2. Markdown as the Source of Truth

The processed Markdown corpus is the authoritative artifact. Embeddings, graphs, and search indexes are derived state and are explicitly disposable. Any operation Memex can do can also be done by another tool reading the same Markdown — Memex is one possible interface to your vault, not its prison.

### 3. Grounded or Refused

Every answer is decomposed into claims, every claim is bound to a cited source chunk, and a verification layer — LLM judgment backed by deterministic checks where that judgment is known to fail — rejects what the evidence doesn't support. When nothing survives, Memex refuses, structurally and visibly. The counterfactual refusal gate is part of the evaluation suite, not a marketing line.

### 4. Small Models, Used Well

Memex commits to running on a single consumer GPU (the reference target is a 12GB card). This forces good engineering: tight prompts, structured outputs, retrieval that earns its tokens, agent loops with budgets, and verification steps that catch model error before it propagates. The constraint produces the discipline.

### 5. Observable at Every Layer

Every parsing decision, enrichment, and agent step is recorded: structured logs, a per-document manifest carrying the model, prompt, and confidence behind each derived artifact, and an on-disk audit event bus. Opt-in self-hosted tracing — off by default, because no-telemetry is the default posture — replays full agent runs: which chunks were pulled, which were rejected, which model produced which output. AI you can't audit is AI you can't trust.

### 6. Composable, Not Captive

Memex exposes its corpus through an MCP server. Any MCP-compatible agent — Claude Code, Cursor, your own — can query the vault. The vault itself is just Markdown, so any tool that reads files reads Memex. There is no proprietary surface area we depend on you depending on. And no *workflow* is captive to one surface: the last terminal-only step — getting a document *into* the vault — is now also a browser drag-and-drop (the exclusive-GPU ingestion mode, ADR-0019), so the CLI is a choice, not a requirement.

---

## The Stack

Memex is opinionated about its stack because constraint at this level is what makes the system viable on consumer hardware. There is no model picker. There is one default configuration that we will keep optimizing as the open-weight landscape evolves. Everything here is open weights, open source, and runs offline.

**Document understanding**

- **PyMuPDF4LLM** as the fast pre-filter for born-digital PDFs — a tiered routing classifier sends clean text-heavy documents down this path at roughly 3× Docling's speed.
- **Docling** (IBM, Apache 2.0) for scans and complex layouts. Handles layout, tables, equations, and outputs structured Markdown directly. CPU-first with optional GPU acceleration.
- **Qwen3-VL-8B-AWQ** (~7.4 GB, served via a short-lived parse-time vLLM process) as the vision-language escalation for pages neither can handle confidently — scanned handwriting, dense diagrams, directed flow/state diagrams, unusual layouts — and the mandatory route for standalone images.
- **Nemotron-Parse** (NVIDIA) as the chart-OCR pass, so the numbers inside a bar chart become queryable data instead of a flattened caption.
- **faster-whisper** for local speech-to-text — audio and video files transcribe into deterministic, timestamped Markdown transcripts that flow through the same pipeline as everything else.

**Reasoning and agents**

- **Qwen3.5-4B** (`cyankiwi/Qwen3.5-4B-AWQ-4bit`, compressed-tensors W4A16, ~6.3 GB live) as the orchestrator and answer model since 2026-06-01 (ADR-0015) — a unified vision-language, hybrid-reasoning model. Strong tool-use, structured-output reliability, multilingual, with an 8,192-token window. *(Was Qwen3-8B-AWQ, retained as the one-flip kill-switch. NB reasoning is suppressed on the strict-guided-JSON grounded path by construction — the orchestrator gains the stronger base + window, not chain-of-thought; the CoT lever's home is the proposed ungrounded expert surface, ADR-0013.)*
- **LangGraph** for state-machine orchestration. Agent loops are explicit graphs with budgets, not free-form ReAct chains.

**Retrieval**

- **EmbeddingGemma 300M** for dense embeddings. Small, multilingual, fast enough to embed a corpus of thousands of pages in minutes on the 4070.
- **bge-reranker-v2-m3** for second-stage reranking on candidates returned by hybrid search.
- **SQLite FTS5** for keyword search; **LanceDB** for vectors; combined via reciprocal rank fusion.

**Knowledge graph**

- **RyuGraph** (embedded, MIT — the maintained fork of Kuzu after the upstream archival; see ADR-0005) for the entity and citation graph. Cypher queries, columnar storage, no server to run.
- **OTTER** (`whoisjones/otter-bi-mmbert`), a compact CPU-side BERT NER, as the recommended entity-extraction backend at enrich time — cleaner typing and roughly double the graph-discovery yield versus LLM extraction.

**Inference**

- **vLLM** as the inference server (OpenAI-compatible API, paged attention, good throughput on Ada-generation GPUs). For lighter use, **Ollama** as a drop-in alternative.

**Observability**

- **structlog** structured logging, per-document manifests, and an on-disk audit event bus — always on, never leaving the machine.
- **Langfuse** (self-hosted, MIT), opt-in and off by default, for replayable traces of agent runs, model calls, retrievals, and tool invocations.

**Interop**

- An **MCP server** exposing the vault as queryable tools (search, retrieve, follow links, summarize, cite). Any MCP client speaks to Memex.

VRAM budget on the reference RTX 4070 (12GB), with the agent and embedding models co-resident: orchestrator ~6.3GB (Qwen3.5-4B at 0.62 util, auto KV), embedder ~600MB, reranker ~600MB, KV cache and overhead. The VLM fallback (Qwen3-VL-8B-AWQ, ~7.4GB — a dedicated vision model, stronger than the 4B's unified vision on hard diagrams, so it is NOT unified) runs only at *parse* time, in its own short-lived vLLM process on the GPU freed by pausing the orchestrator — never co-resident with answering. Inference is sequential by design — the agent is the bottleneck, not parallel decoding.

---

## How Memex Works

Memex is not "OCR plus a chatbot." It is a set of agent loops with bounded scope and explicit verification.

**Ingestion routes by what the document actually is.** A born-digital PDF takes the fast PyMuPDF path; scans and complex layouts route through Docling; pages neither can handle confidently — handwriting, dense diagrams, directed flow charts — escalate to the vision-language model, and charts get a dedicated OCR pass so their numbers become queryable data. Office documents convert through headless LibreOffice and join the PDF pipeline. Audio and video transcribe through local Whisper into timestamped transcripts, and a lecture transcript can be aligned to its slide deck so the spoken explanation and the slide it explains are jointly retrievable — answers cite the slide page and the minute mark. Standalone images — screenshots, photographed pages, exported diagrams — transcribe through the VLM. Source code is stored verbatim and chunked on symbol boundaries, so a query about a function lands on the function. Throughout, tables are extracted as structured tables, equations become LaTeX, and figures are captioned and stored alongside the document.

**Enrichment builds the connective tissue.** Entities are extracted, citations are resolved against the rest of the vault, wikilinks are inserted where confidence is high, and the document's frontmatter is populated with metadata. Every enrichment is recorded in the manifest with its source — model, prompt, confidence — so you can audit or revert any decision later. The resulting entity and citation graph powers deliberate discovery surfaces — related documents ranked by the specificity of what they share, one-hop citation views, entity profiles with their co-occurring concepts — rather than being silently stirred into retrieval.

**Answering is retrieve, rerank, verify — or refuse.** A query pulls hybrid candidates (BM25 plus dense vectors, fused), reranks them with a cross-encoder, and constructs an answer whose every claim cites a source chunk. A verification step then checks each claim against its citation, with deterministic backstops for the cases where LLM judgment is known to fail — fabricated numeric totals, claims grounded only by a name appearing in a list. Table arithmetic runs through SQL with independent recomputation. If the evidence doesn't support an answer, the agent says so — and shows you what it did find. There is no hallucinated confidence. There is no answer without a citation trail.

**Everything is inspectable.** Structured logs, per-document manifests, and an audit event bus record what happened at every stage; opt-in self-hosted tracing replays full agent runs — the retrieved chunks, the rejected candidates, the tokens spent. This is the difference between using AI and trusting AI.

---

## What Makes Memex Different

**Verified, not just retrieved.** Every "chat with your docs" tool retrieves and hopes. Memex retrieves, answers, and then checks the answer — claim by claim, against the cited evidence, with deterministic backstops where the model can't be trusted to grade itself. Refusal is a feature, fabrication is a test failure, and both directions are measured.

**No cloud, no exceptions.** Every other "private RAG" tool has an asterisk somewhere — a managed inference endpoint, a hosted embedding API, telemetry "for product improvement." Memex doesn't. The asterisk is the whole product.

**Markdown out, not lock-in out.** Memex produces a vault that is useful even if you stop using Memex tomorrow. The Markdown is yours. The graph is reproducible. The embeddings are regenerable. There is no migration story because there is nothing proprietary to migrate from.

**Tuned for one rig, orchestrated like an OS.** Memex picks a hardware target (a 12 GB consumer GPU) and tunes for it ruthlessly — pausing and swapping models so an 8B vision model and a 4B agent share one card without ever colliding, adapting placement to live free VRAM, degrading gracefully instead of OOMing. Most local-AI projects either assume an H100 or run so slowly on consumer hardware that they're toys. Memex is built to be usable, every day, on the machine you already own — by scheduling the GPU, not just shrinking the models.

**Agentic from the ground up.** Parsing, enrichment, and querying are all agent loops — bounded, verified, observable, but genuinely agentic. The system can re-attempt, escalate, or refuse. It is not a one-shot pipeline pretending to be intelligent.

**Honestly open source.** Apache or MIT throughout, no contributor license agreement designed to enable a future re-license, no proprietary "pro" tier. If we lose interest, the project is still useful. That is the only meaningful test of an open-source commitment.

---

## Who Memex Is For

**Researchers and graduate students** managing literature reviews across hundreds or thousands of papers, who need semantic search, citation graphs, and cross-paper question-answering without uploading embargoed or sensitive work to a third party.

**Students** building durable study corpora from lecture recordings, slide decks, textbooks, screenshots, and their own annotations — recorded lectures transcribed locally and aligned to their slides, so the spoken explanation and the slide it explains are searchable together. A knowledge base that accumulates across a degree rather than evaporating at the end of each semester.

**Documentation-as-code teams and software engineers** maintaining technical docs, ADRs, RFCs, runbooks, and internal references as Markdown in git — and the codebase itself, ingested verbatim and chunked on symbol boundaries so "where is this implemented?" lands on the exact function. Memex layers semantic search and agentic Q&A on top without changing the source format.

**Prompt library maintainers** organizing growing collections of prompts, evaluations, and outputs into something queryable. Memex treats a prompt library as a first-class document type — every prompt indexed, tagged, linked to its evaluations, and retrievable by intent rather than filename.

**Independent technical writers, archivists, and analysts** working with material they cannot ship to a cloud service — under embargo, under contract, under regulation, or just under principle.

The original document listed lawyers, doctors, and government workers. Those use cases are real, but they require certifications and audits beyond the scope of an open-source project. Memex serves them by being correct, observable, and local — not by claiming compliance it cannot underwrite.

---

## Success Metrics

We measure success in technical quality, in usability on real hardware, and in whether the project survives long enough to matter.

**Honesty, enforced by evaluation**

- Counterfactual refusal: on questions whose answers are deliberately absent from the corpus, the agent must refuse rather than fabricate. This gate is measured per corpus and must hold before changes ship — and it has held across technical slide decks, financial reports, multilingual documents, lecture transcripts, and source code.
- Citation integrity: every presented claim is bound to a source chunk and survives a verification pass; numeric claims additionally survive a deterministic verbatim-presence check against the cited table.
- Over-refusal tracked as a first-class failure alongside hallucination — a system that refuses when the answer is plainly present is as unreliable as one that fabricates.
- Layout-faithful extraction on standard documents: tables-as-tables, equations-as-LaTeX, headings preserved — audited at the raw-Markdown level, because the parsed `.md` is both the source of truth and the retrieval substrate.

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

**Now** (shipped): the end-to-end local pipeline — tiered parsing (PyMuPDF / Docling / VLM escalation / chart-OCR), audio and video transcription, image and source-code ingestion, hybrid retrieval with reranking, the grounded answering agent with the counterfactual refusal gate, table text-to-SQL with independent recomputation, grounded structured summarization, the entity and citation discovery graph, lecture-to-slide alignment, the opt-in reason-then-ground analysis surfaces, a web UI with browser drag-and-drop ingestion, the MCP server, and always-on systemd deployment.

**Next**: answer-stage correctness for code usage queries — retrieval already lands the right symbol; the answer should too. Transitive citation-chain traversal, once vaults hold citation-linked clusters dense enough to walk. Symbol-aware code chunking beyond Rust. Methodological disagreements between papers as first-class queryable structures.

**After that**: speculative parsing on idle GPU time. Domain plugins maintained by the people who actually work in those domains.

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

**Memex** — *Your documents. Your machine. Your trails. Your truth.*
