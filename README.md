# Memex 📚

> Turn the papers, PDFs, and notes on your hard drive into a knowledge base you can actually talk to — without sending a single byte to the cloud.

Memex reads your documents, builds you a Markdown library of what's inside them, and lets you ask questions that get grounded answers with citations. Everything runs on your laptop. 🏠

---

## 🤔 What it's for

You have:
- 🗂️ A folder of academic papers, datasheets, meeting notes, technical manuals.
- 🤷 A vague memory that "someone wrote about X somewhere in there."
- 🚫 A strong preference for not uploading any of it to a third-party service.

Memex helps you:
- 📖 **Read** your documents into clean Markdown (with figures, tables, equations preserved).
- 🔍 **Search** across them with hybrid retrieval (keyword + meaning).
- 💬 **Ask** real questions and get answers that cite the source paragraphs.
- 🔗 **Cross-reference** — Memex builds an entity graph so it knows when two papers talk about the same thing.
- ✏️ **Edit** the Markdown yourself; Memex notices and re-indexes.

---

## ⚡ Quick start

You'll need an **NVIDIA GPU** (RTX 4070 / 12 GB VRAM is the reference rig) and **Python 3.12+**. The heavy lifting — embedding, reranking, parsing low-confidence PDF pages — happens on the GPU. Everything else is local Python.

```sh
# 1️⃣  Install
uv sync --extra models --extra parse

# 2️⃣  Tell Memex where to put your vault
export MEMEX_VAULT_PATH=~/memex-vault

# 3️⃣  Vendor the frontend assets (one-time, downloads HTMX)
./scripts/vendor-frontend.sh

# 4️⃣  Start the local language model in another terminal
./scripts/serve-vllm.sh

# 5️⃣  Talk to Memex
uv run memex ingest path/to/some-paper.pdf
uv run memex ask "What does the paper say about reproducibility?"
```

That's the whole thing. ✨

---

## 🎯 Try the web UI

```sh
uv run memex serve web
# then open http://127.0.0.1:7423
```

You get:
- 💭 An **ask box** with HTMX-driven streaming answers, every claim labelled with its source chunk.
- 📑 A **documents page** with PDF side-by-side preview when you ingested from PDF.
- 🕸️ A **graph view** showing which docs share entities or cite each other.
- ✍️ **Inline correction** — edit a paragraph in the browser, hit save, Memex picks up the change.

The whole UI is dark zinc on near-black with a single blue accent, dense and quiet — built to feel like a desk lamp, not an AI app. 🛋️

---

## 🔌 Plug it into your other tools

Memex ships a **Model Context Protocol** server, so you can use it from Claude Code, Cursor, or any MCP-aware client:

```sh
uv run memex serve mcp --transport stdio
```

Tools the server exposes:
- 🔎 `search(query, k)` — hybrid retrieval over your vault
- ❓ `ask(question)` — full agent with grounded answers
- 📄 `get_document(doc_id)` — fetch the canonical Markdown
- 📚 `list_documents()` — what's in the vault
- 🌐 `get_graph_neighbors(doc_id)` — related documents

Network-facing setup (HTTP transport + bearer-token auth): see [`docs/deploy/mcp-http.md`](docs/deploy/mcp-http.md).

Long-running production deployment (vLLM under systemd / launchd, restart-on-crash, journald log integration): see [`docs/deploy/systemd.md`](docs/deploy/systemd.md) and [`docs/deploy/launchd.md`](docs/deploy/launchd.md).

---

## 🛡️ Privacy & offline guarantees

- 🚫 **No remote endpoints.** The parser worker runs under a `seccomp` filter that blocks every network syscall — even a malicious PDF can't phone home.
- 🪪 **No telemetry.** Memex doesn't ping anyone, ever. Logs stay on disk.
- 🗝️ **Your vault is mode 0700.** Locked down to your user account.
- 🪶 **Local LLM by default.** vLLM runs your chosen 7B/8B model on your GPU. No OpenAI API keys, no Anthropic keys, nothing.
- 📡 **Air-gap test passes.** Pull the ethernet, do everything you'd normally do — Memex doesn't blink.

The only thing that talks to the network is the *initial model download* (a one-time `huggingface-cli download`). After that, you can pull the cable. 🔌

---

## 📂 What's in the vault

```
~/memex-vault/
├── documents/
│   ├── abc12345-paper-on-reflexivity.md      ← the canonical Markdown
│   └── abc12345-paper-on-reflexivity/        ← the original PDF + figures
│       └── source.pdf
└── .memex/
    ├── embeddings.lance     ← vector index (LanceDB)
    ├── search.sqlite        ← keyword index (SQLite FTS5)
    ├── graph.ryu            ← entity + citation graph (RyuGraph)
    └── manifests/*.json     ← audit trail per document
```

**The Markdown files are the source of truth.** 📜 You can edit them, version them with git, sync them across machines — Memex will rebuild every index from them with `memex reindex`.

---

## 🧭 The full picture

| If you want to know… | Read… |
|---|---|
| 🌅 The why (vision + design principles) | [`docs/VISION.md`](docs/VISION.md) |
| 🔧 The how (engineering rules + stack) | [`docs/GUIDELINES.md`](docs/GUIDELINES.md) |
| 🗺️ What's done & what's queued | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| 🏗️ The architecture blueprint | [`docs/IMPLEMENTATION-PLAN.md`](docs/IMPLEMENTATION-PLAN.md) |
| 📐 Why we picked what we picked | [`docs/adr/`](docs/adr/) (ADRs 0001–0006) |

Browse the same content as a navigable site:

```sh
uv sync --extra docs && uv run mkdocs serve
```

Material for MkDocs, dark palette, no Google Fonts (because of course not). 🌒

---

## 🧪 Run the tests

```sh
uv run pytest             # 86 tests, ~5 seconds, no GPU needed
uv run pytest tests/unit  # just the pure-function tests
```

Integration tests fake the heavy I/O (vLLM, Docling, LanceDB) so the full pipeline can run without a GPU. The marquee sandbox test ("can a parsed PDF open a socket?") is skipped on platforms without privileged seccomp. ✅

---

## 📜 License

Apache-2.0. Do as you wish. 🕊️

---

*Built around five principles: local-first by construction, Markdown as the source of truth, small models used well, observable at every layer, composable not captive. None of these are aspirational — every one is enforced by the test suite.* 🌱
