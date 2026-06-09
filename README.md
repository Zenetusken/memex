# Memex 📚

> Turn the papers, PDFs, and notes on your hard drive into a knowledge base you can actually talk to — without sending a single byte to the cloud.

Memex reads your documents, builds you a Markdown library of what's inside them, and lets you ask questions that get grounded answers with citations. Everything runs on your laptop. 🏠

```
You: ── PDFs, notes, papers ──►  Memex  ──►  Markdown vault + indexes + answers
                                  │
                            stays on your disk
```

---

## 🤔 What it's for

You have:
- 🗂️ A folder of academic papers, datasheets, meeting notes, technical manuals.
- 🤷 A vague memory that "someone wrote about X somewhere in there."
- 🚫 A strong preference for not uploading any of it to a third-party service.

Memex helps you:
- 📖 **Read** your documents into clean Markdown (with figures, tables, equations preserved) — PDFs, Office files, scans, standalone **images** (screenshots, diagrams, photographed pages), and **audio / video recordings** (lectures, meetings) transcribed to a timestamped transcript.
- 🔍 **Search** across them with hybrid retrieval (BM25 + dense + cross-encoder rerank).
- 💬 **Ask** real questions and get answers that cite the source paragraphs — or *refuse* when there isn't enough grounding (refusal is a first-class outcome).
- 🔗 **Cross-reference** through an entity graph that links docs sharing topics, citations, or named entities.
- ✏️ **Edit** the Markdown yourself; Memex notices and re-indexes.
- 🔌 **Plug in** to Claude Code, Cursor, or any MCP-aware client.

---

## 🖥️ System requirements

| Component | Minimum | Reference rig |
|---|---|---|
| GPU | NVIDIA, **8 GB VRAM**, Ada Lovelace (sm_89) or newer | RTX 4070 12 GB |
| CUDA driver | R570+ (cu129 wheels) | — |
| CPU | Any modern x86_64 (8+ cores comfortable) | — |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB for models + your vault | — |
| OS | **Linux** (Ubuntu 22.04+, Pop!_OS, Fedora 39+, Arch). macOS is dev-tier only — the CUDA-only stack doesn't apply. Windows is unsupported. | Pop!_OS 22.04 |
| Python | 3.12+ | — |
| LibreOffice | _Optional_ — only to ingest Office docs (`.pptx`/`.docx`/`.xlsx`); `soffice` must be on PATH | system package (`apt install libreoffice` / `dnf install libreoffice`) |

Per ADR-0001 the project doesn't ship a CPU fallback. If `torch.cuda.is_available()` is False, startup fails fast with a clear message.

The **default 12 GB-tier orchestrator is `cyankiwi/Qwen3.5-4B-AWQ-4bit`** — a unified vision-language, hybrid-reasoning model (compressed-tensors W4A16) adopted 2026-06-01 ([ADR-0015](docs/adr/0015-qwen35-4b-unified-orchestrator.md); the full re-baseline held every HARD gate). The legacy `Qwen3-8B-AWQ` remains the one-flip kill-switch (revert `models.orchestrator` + `memex daemon restart` — zero re-indexing). The parse-time diagram **doc-VLM stays `Qwen3-VL-8B`** (unifying it onto the 4B was attempted + reverted — it regressed). See [`docs/deploy/hardware-tiers.md`](docs/deploy/hardware-tiers.md) for the env-var matrix + the eval-verified quality numbers.

---

## ⚡ Install

```sh
# 1. Clone + sync
git clone https://github.com/Zenetusken/memex.git ~/project/Doc_Flo
cd ~/project/Doc_Flo
uv sync --extra models --extra parse --extra serve --extra audio

# 2. (optional, recommended) Pre-fetch the HuggingFace models so first
# boot doesn't hammer the network. ~13 GB on disk.
uv run python scripts/download-models.py

# 3. Vendor the HTMX + Tailwind subset (one-time; SHA-384 verified).
./scripts/vendor-frontend.sh

# 4. Pick your vault directory (or accept the default ~/.memex/vault).
export MEMEX_VAULT_PATH=~/memex-vault
mkdir -p "$MEMEX_VAULT_PATH"
```

That installs the four Memex extras:

| Extra | What it brings | When you need it |
|---|---|---|
| `models` | torch + transformers + sentence-transformers + accelerate (all cu129) | Always |
| `parse` | Docling + PyMuPDF4LLM + pyseccomp + pypdfium2 | Always (PDF / DOCX / PPTX ingest) |
| `serve` | vLLM (cu129 wheels) | If you're running the inference daemon locally (almost always) |
| `audio` | faster-whisper (CTranslate2 — ships its own CUDA libs, ABI-independent of the torch/vLLM wheels) | Ingesting **audio / class-lecture** files (MP3/WAV/M4A/FLAC/OGG/…) **and native video** (MP4/MOV/WEBM/MKV/… — the audio track is transcribed) — the ASR transcription route (ADR-0017). Also set `MEMEX_MODELS__ASR` to a Whisper build. |
| `dev` | pytest + ruff + pyright + hypothesis + pre-commit | Contributing or running the test suite |
| `eval` | ragas + jiwer | Running `memex eval` against a query set |
| `docs` | mkdocs + mkdocs-material | Building the documentation site |

---

## 🚦 Pick your deployment pattern

There are three supported ways to run Memex. Pick one — they don't conflict, but **the always-on systemd flow is the recommended production setup** and what the rest of this README assumes unless noted.

### A) Always-on services (recommended on Linux)

The whole stack — vLLM + web UI + MCP server + vault watcher — runs as user-level systemd services. Boots with your session, restart-on-crash, log rotation via journald, **no terminal babysitting**.

```sh
# One-time install
mkdir -p ~/.config/systemd/user ~/.config/memex ~/.local/state/memex
cp docs/deploy/memex-vllm.service docs/deploy/memex-web.service \
   docs/deploy/memex-mcp.service  docs/deploy/memex-watch.service \
   ~/.config/systemd/user/
cp docs/deploy/memex-vllm.env docs/deploy/memex-web.env \
   docs/deploy/memex-mcp.env  docs/deploy/memex-watch.env \
   ~/.config/memex/

# Generate an MCP bearer token (skip if you'll only use stdio MCP)
echo "MEMEX_MCP__AUTH_TOKEN=$(uv run memex mcp generate-token)" \
  >> ~/.config/memex/memex-mcp.env

# Edit each .service file: change `WorkingDirectory=` and `ExecStart=`
# to match where you cloned Memex. The default templates assume
# %h/project/Doc_Flo — adjust if needed.

systemctl --user daemon-reload
systemctl --user enable --now memex-vllm.service memex-web.service \
                              memex-mcp.service  memex-watch.service
```

Verify:

```sh
systemctl --user status memex-vllm memex-web memex-mcp memex-watch --no-pager
journalctl --user -u memex-vllm -u memex-web -u memex-mcp -u memex-watch -f
```

After this, you never start anything by hand again. `memex ask`, `memex ingest`, the web UI at <http://127.0.0.1:7423>, and the MCP HTTP endpoint at <http://127.0.0.1:7424> are all permanently available.

Full guide: [`docs/deploy/systemd.md`](docs/deploy/systemd.md).
macOS equivalent (launchd plists): [`docs/deploy/launchd.md`](docs/deploy/launchd.md).

### B) One-shot from two terminals (demo / dev)

If you just want to try Memex without installing services:

```sh
# Terminal 1 — start the inference daemon
./scripts/serve-vllm.sh
# wait ~40s for "Application startup complete."

# Terminal 2 — use Memex
uv run memex ingest path/to/some-paper.pdf
uv run memex ask "What does the paper say about reproducibility?"
```

This is what the original quickstart looked like; it works, but you lose vLLM the moment you close terminal 1.

### C) Detached background (no systemd)

Memex ships a small wrapper that detaches vLLM via `nohup`-style session leadership:

```sh
uv run memex daemon start
# vLLM is now running in the background; check status:
uv run memex daemon status
# stop it:
uv run memex daemon stop
```

This is intermediate between (A) and (B) — survives terminal close, doesn't survive logout/reboot, no restart-on-crash. Useful on a Mac dev box where launchd feels heavy. Don't mix this with (A); they'd race.

---

## 💬 Common workflows

After installing per **Pattern A** above, these are the everyday commands:

### Ingest documents

```sh
uv run memex ingest some-paper.pdf                    # one file
uv run memex ingest *.pdf                             # batch
uv run memex ingest ~/Downloads/papers/               # whole folder
```

Each PDF goes through: validation → ingest copy → parse (PyMuPDF for born-digital, Docling for scans) → index (chunks → LanceDB + FTS5) → enrich (entity extraction + citation/graph edges — entities via a pluggable NER backend [the LLM by default, the OTTER BERT-NER recommended; see the enrich-NER config below], citations via the LLM). All inline, all logged.

For scanned content add `MEMEX_PARSE_DOCLING_OCR=1`; for born-digital PDFs (PowerPoint exports, LaTeX, Word) leave it off — the PyMuPDF pre-filter handles those at ~3× Docling's speed.

**Office documents** (`.pptx`/`.docx`/`.xlsx` + their ODF cousins) are first converted to PDF via headless LibreOffice, then run through the normal PDF pipeline — so a diagram-heavy slide deck's figures flow through the VLM diagram-transcription + chart-OCR passes (enable the VLM with `MEMEX_PARSE__DISABLE_VLM=false` for diagram-rich decks). The converted PDF is cached per document and reused on re-parse. LibreOffice must be installed (`soffice` on PATH). On the 12 GB tier `memex ingest`/`index`/`reindex` pause vLLM for the duration so the embedder isn't starved by a co-resident vLLM — no manual `MEMEX_RERANK_BATCH_SIZE` or daemon juggling needed. The VLM is **Qwen3-VL-8B-AWQ**, run as a short-lived vLLM process *during parse only* (the orchestrator is paused to free the GPU); it transcribes directed flow/state diagrams correctly where the older Qwen2.5-VL flattened them into a list. Tune via `MEMEX_MODELS__VLM_SERVING` (`vllm` | `transformers`) + `MEMEX_MODELS__VLM_SERVE__*` (port / `gpu_memory_utilization` / `max_model_len`) — see [`docs/specs/vlm-vllm-serving.md`](docs/specs/vlm-vllm-serving.md).

**Audio + video recordings** (audio `.mp3`/`.wav`/`.m4a`/`.flac`/`.ogg`/… and native video `.mp4`/`.mov`/`.webm`/`.mkv`/…) ingest through a speech-to-text route (ADR-0017): faster-whisper transcribes the media to a deterministic `## [mm:ss]` Markdown transcript (segments coalesced into ~30 s blocks), which then flows through the normal parse/index/answer path. A transcript chunk carries a `time_range` so the web UI shows time chips and answers cite the moment. Needs the `audio` extra (included in the default `uv sync` above) and a Whisper build set via `MEMEX_MODELS__ASR` (reference: `large-v3-turbo`). A lecture transcript can also be **aligned to its slide deck** via `memex link-slides` (ADR-0018), making the slides and the spoken commentary jointly groundable. Spec: [`docs/specs/audio-asr-route.md`](docs/specs/audio-asr-route.md).

**Standalone images** (`.png`/`.jpg`/`.jpeg`/`.webp`/`.bmp`/`.tif`/`.tiff`/`.gif`) ingest as a one-page scan (ADR-0020): the image is wrapped into a cached 1-page PDF and run through the VLM transcription route, so a screenshot, infographic, exported topology diagram, or photographed page becomes searchable + grounded-answerable. The VLM is **mandatory** for images (an image has no text layer to read — the same precedent as audio always running ASR), so `memex ingest diagram.png` works out of the box even with the default `MEMEX_PARSE__DISABLE_VLM=true`. An unreadable image is HARD-gate-safe either way: it transcribes to nothing (refused, no document written) or the VLM returns an honest "the image is blank" caption — a thin document that then refuses every substantive query; never a junk document with fabricated content. HEIC/AVIF are not yet accepted (they need a separate decode dependency). Spec: [`docs/specs/image-ingestion.md`](docs/specs/image-ingestion.md).

**Source code** (`.rs`/`.py`/`.ts`/`.go`/`.js`/… — the `CODE_SUFFIXES` set) ingests as **documents** (ADR-0021), not through the prose layout model that mangles aligned code into pipe-tables. A code file is stored **verbatim** (the canonical `.md` is the literal source) and chunked on **symbol boundaries** — for Rust, one chunk per `fn` / `struct` / `impl` method — so "where is `FunctionCallOutputPayload` serialized?" lands on the exact symbol rather than a size-budgeted fragment. The web UI renders a code document *as code* (`source · rust`, no heading/wikilink transforms — a `# comment` is not a heading). `scripts/ingest_codebase.py` walks a whole repository (titles are repo-relative paths; `.git`/`target`/`node_modules` skipped). Symbol-aware chunking is **Rust-first** in v1; other languages ingest verbatim and chunk by size for now. Spec: [`docs/specs/code-chunking.md`](docs/specs/code-chunking.md).

**Or ingest from the browser** — the web UI has an **Add document** page: drag/drop or pick a file and it runs the *whole* pipeline (parse → VLM/chart-OCR/ASR → index → enrich) with chat-style live progress and a real-time VRAM panel, then lands the doc fully searchable + browsable. Ingestion is an **exclusive-GPU mode**: while a document is being consumed the answering surfaces pause (you can still browse everything already ingested), so all VRAM goes to the pipeline. No terminal needed.

### Ask grounded questions

```sh
uv run memex ask "What does the paper say about ablations?"
uv run memex ask "Compare how Smith and Tan handle reflexivity"
uv run memex ask "Has anyone discussed CUDA-graph capture overhead?"
```

Output is JSON on a pipe, rich tables in a terminal. Every `claim` carries a `source_chunk_id`; the agent **refuses** when chunks don't ground the answer (returns `answered: false` + `refusal_reason`).

Numeric/aggregate/superlative questions over tables ("total fees paid to all directors", "which segment had the highest revenue") run a structured **text-to-SQL** pass over a per-vault table store (`tables.sqlite`, built at index time) instead of LLM arithmetic over a truncated markdown slice. The no-hallucination gate holds by construction: row-returning SQL injects only verbatim cells, and a computed aggregate ships **only** when an independent Python recompute over the original cells agrees — otherwise it falls back to normal retrieval (which refuses if it can't ground). See [`docs/specs/table-sql.md`](docs/specs/table-sql.md) and [ADR-0014](docs/adr/0014-text-to-sql-robustness-safety.md).

### Summarize a whole document

```sh
uv run memex summarize 2f96ae1c-some-paper                       # standard detail
uv run memex summarize 2f96ae1c-some-paper --detail detailed     # longer
uv run memex summarize 2f96ae1c-some-paper --detail report       # multi-paragraph report
uv run memex summarize 2f96ae1c-some-paper -i "focus on the method"
```

Produces a **structured, grounded** summary (ADR-0008): an abstract + cited key-points + per-section digests, built by a map-reduce over the document's indexed chunks. Every key-point is grounded to a source chunk or dropped, and a document with nothing groundable **refuses** — the same no-hallucination gate the answering agent uses, extended to summaries. `--detail` (`brief`/`standard`/`detailed`/`report`) tunes length — `report` produces a coherent **multi-paragraph** body via a hierarchical reduce (one bounded paragraph per section-group, stitched with a deterministic cross-paragraph dedup pass; ADR-0010); `--token-budget` caps the work on very long docs. Quality is identical whether you're in `fast` or `full` co-residence mode (the strategy is chosen by the document, not the mode). Also available as the **Summarize** button on the web UI document view, and the MCP `summarize` tool.

### Switch the co-residence mode (speed vs. context)

On a 12 GB card the orchestrator's context window and the GPU-resident reranker compete for ~3 GB of swing VRAM. A *mode* is a named bundle of that tradeoff (ADR-0007):

```sh
uv run memex mode show              # the active profile + VRAM estimate
uv run memex mode set auto          # the default — adapts placement to live free-VRAM
uv run memex mode set fast          # GPU reranker, 6 K context, ~14 s/ask
uv run memex mode set full          # reranker→CPU (fallback for concurrent-GPU / long eval), 24 K context
```

**`auto` is the default mode** (ADR-0007 P4.4): it reads *live* free-VRAM at load and adapts — the embedder always stays on the GPU, and the reranker co-resides on the GPU when there's ≥2 GB free, else falls back to CPU (graceful, never an OOM). This is **correctness-neutral** — the reranker order is byte-identical CPU vs GPU, so placement only changes latency, never the answer — so it works optimally out of the box with zero manual config. `fast` is low-latency top-k RAG; `full` frees the GPU into a ~24 K orchestrator window (the reranker moves to CPU, ~20–30 s/query). On the 4B orchestrator the reranker co-residing on the GPU (the common `auto`/`fast` case for single-process `ask`/`chat`/`expert`/`bridge`) is ~70–90× faster than CPU, quality-identical, and fits with ~3 GB headroom; `full` moves it to CPU only when you need the wide window or are sharing the GPU with another process. `memex mode set` restarts the daemon-managed orchestrator; set `MEMEX_MODELS__CO_RESIDENCE_MODE` + restart `memex serve web` to also move the retrieval models. The full matrix + the eval numbers are in [`docs/deploy/hardware-tiers.md`](docs/deploy/hardware-tiers.md); the design is [ADR-0007](docs/adr/0007-co-residence-resource-modes.md).

### Browse the vault

```sh
# CLI listing
uv run memex list documents

# Or the web UI
open http://127.0.0.1:7423
```

The web UI gives you a documents list, **side-by-side preview of the source PDF** (server-rendered page images, lazy-loaded — works in every browser regardless of the "download PDFs" setting), a server-rendered "connections" view (related documents ranked by shared-entity specificity, grouped under the bridging concepts — no client-side graph library), and an inline edit-then-save flow (with conflict detection if someone else changed the doc since you started editing — see [`docs/deploy/mcp-http.md`](docs/deploy/mcp-http.md) for the 409 conflict surface). Long agent/summarizer runs surface **live progress** via an HTMX long-poll widget (no SSE / no new JS) — the per-section counter ticks on Summarize, the agent's node-by-node phase advances on Ask; the answer + summary panels label sources by **document title › section** (the raw `docid#hash` is kept as a tooltip), so a long deck's claims read as English rather than hashes.

### Edit a Markdown document

In the web UI, click **edit**, change the text, hit **save**. If the file changed under you (because of a parallel ingest, watcher reaction, or another tab) you get a 409 panel with a unified diff and **"discard mine & reload"** + **"overwrite anyway"** buttons.

Outside the web UI, just edit the file:

```sh
$EDITOR ~/.memex/vault/documents/2f96ae1c-some-paper.md
```

The watcher service notices the sha change, re-enriches, and re-indexes the doc within seconds. No manual `memex reindex` needed.

### Rename a document's title

A document's title is pure metadata — it isn't part of the embedded text or the chunk IDs — so renaming it never re-embeds anything:

```sh
uv run memex retitle 2f96ae1c-some-paper "A Clean Human Title"   # explicit
uv run memex retitle 2f96ae1c-some-paper --derive                # from the source filename
```

This rewrites the frontmatter title and fans it out to the FTS, vector, and graph indexes in one cheap, GPU-free pass (it doubles as a way to repair a stale title). The web UI exposes the same thing: click **rename** next to the document title, edit inline, save. A clean title also helps cross-document citation resolution, which scores against other docs' titles.

### Remove a document

```sh
uv run memex remove 2f96ae1c-some-paper        # prompts for confirmation
uv run memex remove 2f96ae1c-some-paper --yes  # skip the prompt
```

Drops the document everywhere — its canonical Markdown, asset dir, manifest, and all derived index state (vector, FTS, tables, graph). Irreversible (the Markdown is the source of truth), so re-add the original source to restore it.

### Explore connections

```sh
uv run memex related -d 2f96ae1c-some-paper        # documents related to this one
uv run memex cites -d 2f96ae1c-some-paper          # references: what it cites + what cites it
uv run memex entity "DNS"                          # everything across the corpus about an entity
```

`memex related` surfaces the documents most related to this one through the entity graph — ranked by the **specificity** of the entities they share, so a sibling that shares one rare concept beats one that shares five generic terms (a near-universal entity, or an incidental person/place name, is filtered out). Each result shows the connecting entities — the *why*. Also a "Related documents" section on the web UI document view and the MCP `related_documents` tool.

`memex cites` is the citation view: a document's 1-hop **References** — what it cites and what cites it (the resolved in-vault citations, with the citation surface form). It's the honest one-hop surface; transitive citation-chain following is deferred until the vault holds a citation-linked cluster dense enough to traverse. Also a "References" section on the web UI document view and the MCP `document_citations` tool.

`memex entity <name>` is the entity-centric view: given an entity name it returns its graph **profile** — canonical kind(s), how many documents mention it, the documents themselves, and the **co-occurring concepts** (ranked by the same specificity filter) — plus representative **passages** from full-text search of those documents. It also bridges **acronym ↔ expansion**: looking up `DNS` surfaces an "Also see → Domain Name System" link (and vice versa) when both forms exist as separate entities — a deterministic initialism match, suggested as a link, never a silent identity merge. An unknown name falls back honestly to a whole-corpus text search (with a "Did you mean?" if a bridge exists). Also the web UI `/entity` page (co-occurring concepts and bridges are links, so you can walk the graph) and the MCP `entity_overview` tool. *Documents and co-occurring concepts come from the entity graph; quoted passages come from full-text search.*

Both are discovery surfaces over the graph (ADR-0011), *not* in the `/ask` retrieval path — a measured audit showed 1-hop graph expansion adds nothing to answering at this corpus scale, so they're deliberate discovery features, not passive recall-boosting.

### Reason over the vault (ungrounded analysis)

The everyday answer path **refuses** when your vault doesn't ground an answer — by design. For genuinely analytical questions (synthesis, advisory, "what would you expect…"), Memex has two **opt-in, off-by-default** surfaces that reason from the model's own knowledge (enable with `MEMEX_AGENTS__EXPERT_MODE_ENABLED=true`):

```sh
uv run memex expert "How would these two designs trade off under load?"   # ungrounded analysis (ADR-0013)
uv run memex bridge "Compare the security postures across my NIST docs"    # reason, THEN ground each claim
```

By default `memex bridge` is the **reason-then-ground** path (ADR-0016): it reasons over the retrieved evidence, extracts the discrete claims that reasoning made, and runs each one through the *same* vault-grounding check a normal `ask` uses — presenting only the survivors as cited, with the rest left inside a clearly-labelled "ungrounded analysis" block. From a refused `ask` in the web UI you get a one-click **"Reason & verify this →"** button that re-runs the question through the bridge over the same documents (you choose it; it never happens automatically) — and when that reasoning grounds a subset that actually answers the question, the bridge presents those vault-verified claims AS the answer (a distinct **"Reasoned, then grounded"** surface with a VERIFIED tag, the ungrounded reasoning fenced in a collapsed details block); otherwise it falls back to the labelled-analysis view. On the CLI the same present-as-answer behaviour is the `memex bridge --answer` flag; the standalone **Analysis** tab / bare `memex bridge` stays verify-only labelled analysis. The **Analysis** tab also carries the same document scope-picker as Ask, so you can focus a fresh analysis on a few documents (and it shows you which ones it scoped to). Every Analysis result also shows a navigable **"Retrieved from your vault"** section listing the documents the model was shown (`title › section · page`, linking into the doc viewer) — labelled as context, *not* grounding cites — so even a zero-grounded analysis lets you open the named docs and check the vault yourself (the CLI prints the parity "Retrieved from your vault …" line). Both surfaces reason from model knowledge, not a verified vault lookup, and are labelled as such. In the web UI they're the **Expert** and **Analysis** tabs (visible only when the feature is enabled). They are deliberately **not** exposed over MCP. See [`docs/specs/grounded-agentic-chat.md`](docs/specs/grounded-agentic-chat.md) §11 + [ADR-0013](docs/adr/0013-ungrounded-reasoning-expert-mode.md) / [ADR-0016](docs/adr/0016-reason-then-ground-bridge.md).

### Update to a newer Memex

```sh
uv run memex upgrade
```

Three steps in order: `git pull --ff-only` → `uv sync --extra models --extra parse --extra serve --extra audio` → `systemctl --user restart` of every installed `memex-*.service`. Refuses if your tree has uncommitted changes (run `git stash` yourself if you want to keep them). vLLM's restart blocks ~30 s for the `Type=notify` readiness gate; the CLI prints a note so you don't mistake it for a hang.

Flags: `--dry-run` previews, `--no-restart` for Pattern B / Pattern C boxes (no systemd), `--skip-sync` for git-pull-and-restart only.

### Plug Memex into Claude Code / Cursor / Claude Desktop

Memex is an MCP server. From a desktop MCP client, point at the stdio transport:

```json
{
  "mcpServers": {
    "memex": {
      "command": "uv",
      "args": ["run", "memex", "serve", "mcp", "--transport", "stdio"],
      "cwd": "/home/YOU/project/Doc_Flo"
    }
  }
}
```

For remote / network access, point at the HTTP transport with the bearer token from your `memex-mcp.env`:

```json
{
  "mcpServers": {
    "memex": {
      "url": "http://YOUR-HOST:7424/mcp/",
      "headers": { "Authorization": "Bearer ze1Q9k…ZW" }
    }
  }
}
```

Ten tools are exposed:
- 🔎 `search(query, k)` — hybrid retrieval over the vault
- ❓ `ask(question, scope_doc_ids?, scope_set?)` — full grounded answering agent (the answer also carries `related_documents`: graph neighbours of the cited docs)
- 📝 `summarize(doc_id, instruction?, detail?)` — structured grounded document summary
- 📄 `get_document(doc_id)` — canonical markdown + frontmatter
- 📚 `list_documents()` — every doc in the vault
- 📌 `list_scope_sets()` — every saved document scope set
- 🌐 `get_graph_neighbors(doc_id)` — one-hop entity neighbors
- 🔗 `related_documents(doc_id)` — related docs ranked by shared-entity specificity (discovery)
- 📎 `document_citations(doc_id)` — 1-hop "References": what this doc cites + what cites it (discovery)
- 🧭 `entity_overview(name)` — an entity's profile (kind(s), mentioning docs, co-occurring concepts) + passages (discovery)

### Inspect health + breakers

```sh
uv run memex doctor              # checks GPU, vLLM, vault, breaker state
journalctl --user -u memex-vllm -u memex-web -u memex-mcp -u memex-watch -f
```

---

## 🛠️ Configuration reference

Every knob is set via environment variable or `~/.config/memex/config.toml`. Env-var prefix is `MEMEX_`; nested keys use `__` as the separator (`MEMEX_INFERENCE__BASE_URL`).

### Core paths + transport

| Variable | Default | Purpose |
|---|---|---|
| `MEMEX_VAULT_PATH` | `~/.memex/vault` (or `$XDG_DATA_HOME/memex/vault`) | Where the canonical Markdown + indexes live. Mode 0700. |
| `MEMEX_INFERENCE__BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint the agent calls. Override if vLLM is on a different host/port. |

### vLLM daemon (read by `scripts/serve-vllm.sh`)

| Variable | Default | Purpose |
|---|---|---|
| `MEMEX_VLLM_MODEL` | _(derived from `models.orchestrator`)_ | The daemon's `orchestrator_serve_env` bridge (ADR-0015) exports this from `models.orchestrator` (default `cyankiwi/Qwen3.5-4B-AWQ-4bit`), so you normally set `MEMEX_MODELS__ORCHESTRATOR`, not this directly. Kill-switch: `MEMEX_MODELS__ORCHESTRATOR=Qwen/Qwen3-8B-AWQ`. |
| `MEMEX_VLLM_HOST` | `127.0.0.1` | Bind host. |
| `MEMEX_VLLM_PORT` | `8000` | Bind port. |
| `MEMEX_VLLM_QUANTIZATION` | _(model-keyed; bridge-derived)_ | Quantization kernel. The default 4B is compressed-tensors → **empty** (`""`); the 8B kill-switch uses `awq_marlin`. `serve-vllm.sh` keys this off the model and the daemon bridge exports it from config (ADR-0015). Set `awq` for the legacy kernel. |
| `MEMEX_VLLM_MAX_MODEL_LEN` | `8192` (4B) / `6144` (8B) | Max sequence length, model-keyed in `serve-vllm.sh`. The 4B default is 8192; the 8B kill-switch is 6144 (sized to fit the production answer prompt at `top_k=5`, chunks truncated to 1800 chars). |
| `MEMEX_VLLM_GPU_FRACTION` | `0.62` (4B) / `0.72` (8B) | 12 GB-rig util, model-keyed. The 4B (auto KV) fits at `0.62`; the 8B kill-switch uses `0.72`. **Drop ~0.04 when chart-OCR is enabled** (default since 2026-05-23) — embedder + reranker + chart-OCR slot need headroom. **8 GB tier: `0.50`** for the smaller orchestrator. |
| `MEMEX_VLLM_KV_CACHE_DTYPE` | `auto` (4B) / `fp8_e5m2` (8B) | Model-keyed. The 4B is an **fp8 checkpoint → `auto`** (vLLM blocks fp8 KV cache + fp8 weights at startup); the 8B AWQ-int4 kill-switch uses `fp8_e5m2` (halves KV-cache memory). |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index. |
| `MEMEX_VLLM_EAGER` | _(unset)_ | Set to anything to disable CUDA-graph compilation (slower decode, faster startup). |

### Parse stage

| Variable | Default | Purpose |
|---|---|---|
| `MEMEX_PARSE__PYMUPDF_ENABLED` | `true` | When `false`, all PDFs route through Docling (the pre-PyMuPDF behaviour). |
| `MEMEX_PARSE__PYMUPDF_MIN_CONFIDENCE` | `0.5` | Confidence threshold for trusting the PyMuPDF extraction. Lower to be more eager; raise to be conservative. |
| `MEMEX_PARSE__PYMUPDF_MIXED_CONTENT_IMAGE_AREA_THRESHOLD` | `0.35` | Image-area share that triggers the mixed-content (force-OCR) routing path. Lower → more aggressive OCR. |
| `MEMEX_PARSE__PYMUPDF_MIXED_CONTENT_MIN_IMAGE_HEAVY_PAGES` | `0.30` | Companion gate; both image-area AND image-heavy-pages must trip for mixed-content to fire. |
| `MEMEX_PARSE__FORCE_DOCLING` | `false` | When `true`, the PyMuPDF classifier is bypassed and every PDF routes directly to Docling. Per-call: `memex parse <doc-id> --force-docling`. Cost: Docling is ~10× slower than PyMuPDF on text-heavy docs. Use to force chart-OCR onto born-digital text-heavy PDFs the classifier would otherwise send to the fast PyMuPDF path. |
| `MEMEX_PARSE__DISABLE_CHART_OCR` | `false` | Disables the chart-OCR pass over Docling figures. Default enabled with `nvidia/NVIDIA-Nemotron-Parse-v1.2` since the 2026-05-23 P3.3-c shootout + v7 fix arc (Q08 "On Time 22 / Late 8" + Q31 "nvmath-python 4 principles" + 5 other chart-content REF→ANS flips across 3 corpora). |
| `MEMEX_MODELS__CHART_OCR` | `nvidia/NVIDIA-Nemotron-Parse-v1.2` | Chart-OCR backend HF id. Alternatives in tree: `khhuang/chart-to-table` (UniChart, smaller + faster but −1 ANS on prose-heavy corpora), `google/deplot` (legacy P3.3 v6 default; same −1 ANS), `kppkkp/OneChart` (CUDA-asserts on OOD imagery — keep for chart-heavy-only re-attempts). |
| `MEMEX_PARSE_DOCLING_OCR` | `0` | Set to `1` to force OCR on Docling. Default off (born-digital PDFs don't benefit; +10× wall time for no answer improvement on the canonical test deck). |

### Enrich stage

| Variable | Default | Purpose |
|---|---|---|
| `MEMEX_AGENTS__ENRICH_NER_BACKEND` | `llm` | Entity-extraction backend at enrich. `llm` = the orchestrator (now Qwen3.5-4B, ADR-0015) extracts entities; `otter` = the span NER `whoisjones/otter-bi-mmbert` (a BERT, runs CPU-side, lazy-loaded once). OTTER types entities far more cleanly (tool/method vs the LLM's generic concept-dump) and roughly doubles graph-discovery yield (+103% `related_documents` on the reference 47-doc vault). **Enrich-graph-only** — citation extraction and the answer path stay on the LLM, so the no-hallucination gate is untouched. Switching backends needs a re-`enrich` (or `reindex`) of existing docs. See [`docs/specs/ner-enrich.md`](docs/specs/ner-enrich.md) + [ADR-0012](docs/adr/0012-otter-bert-ner-enrich-backend.md). |
| `MEMEX_AGENTS__ENRICH_NER_MODEL` | `whoisjones/otter-bi-mmbert` | HF id for the OTTER NER (consulted only when `backend=otter`). |
| `MEMEX_AGENTS__ENRICH_NER_THRESHOLD` | `0.05` | OTTER span-confidence floor — the master knob. The model card's 0.1 strangles recall; 0.05 is the A/B-validated sweet spot. |
| `MEMEX_AGENTS__ENRICH_NER_LABELS` | `union` | OTTER label set: `generic` / `domain` / `union`. `union` (both) is the A/B winner — resolves corpus-dependence. |
| `MEMEX_AGENTS__ENRICH_NER_DEVICE` | `cpu` | `cpu` or `cuda`. CPU by default; `cuda` is viable during the CLI enrich's pause-vLLM window. |

### Index stage

| Variable | Default | Purpose |
|---|---|---|
| `MEMEX_INDEX__CHUNK_TARGET_TOKENS` | `400` | Word-count target for chunks. ≈ 520 transformer tokens. Bump to 600 on rigs with `max-model-len >= 8192`. |
| `MEMEX_INDEX__CHUNK_OVERLAP_TOKENS` | `60` | Word-count overlap between chunks. Scales with target. |
| `MEMEX_INDEX_EMBED_BATCH` | `32` | Embedder batch size. Push higher on bigger GPUs for throughput. |
| `MEMEX_EMBED_NATIVE_PROMPTS` | `1` | Use EmbeddingGemma's native `task:`/`title:` prompts (its trained, in-distribution usage). **Changing this REQUIRES a `memex reindex`** — queries are embedded with the live setting while the index stores vectors from whatever setting built it; the index records its recipe (and a reindex auto-re-embeds on mismatch), but querying a stale index in the window before re-embed silently degrades dense retrieval. Set to `0` only to A/B against bare embedding, and reindex after. |

### Retrieve stage

| Variable | Default | Purpose |
|---|---|---|
| `MEMEX_RERANK_BATCH_SIZE` | `8` | bge-reranker pair-batch size. Empirical 12 GB-rig floor; bump to 32–64 on bigger rigs / smaller orchestrators. On a CUDA OOM the reranker auto-retries once at batch 1, so a too-large value degrades gracefully instead of crashing mid-answer — set it low upfront on a constrained rig only to skip the wasted first attempt. |
| `MEMEX_RERANK_TOP_K` | `5` | Reranked chunks fed to the agent. Sized to fit the answer prompt's chunk truncate (1800 chars) within `max-model-len=6144`. Drop to 4 if chunks are unusually dense; bump to 8+ with larger context windows. |
| `MEMEX_MODELS__RERANKER_BACKEND` | `cross_encoder` | `qwen3` swaps in Qwen3-Reranker-0.6B. **Quality A/B verdict 2026-05-21**: `cross_encoder` (bge-reranker-v2-m3) wins clearly on the slide-decks benchmark (median ANS=4 vs qwen3's 0) — Qwen3-Reranker ranks thematically-general chunks above the literal-answer chunk. Stay on `cross_encoder` unless your corpus favours topical similarity over fact-extraction. |

### Network / security

| Variable | Default | Purpose |
|---|---|---|
| `MEMEX_MCP__AUTH_TOKEN` | _(unset)_ | When set, the MCP HTTP transport requires `Authorization: Bearer <token>`. When unset, non-loopback binds are refused at startup. Generate via `uv run memex mcp generate-token`. |

### Observability

| Variable | Default | Purpose |
|---|---|---|
| `MEMEX_OBSERVABILITY__LANGFUSE_ENABLED` | `false` | Off by default (local-first / no-telemetry). Set true + both keys below to enable tracing. |
| `MEMEX_OBSERVABILITY__LANGFUSE_PUBLIC_KEY` | _(unset)_ | Self-hosted Langfuse public key. |
| `MEMEX_OBSERVABILITY__LANGFUSE_SECRET_KEY` | _(unset)_ | Self-hosted Langfuse secret key. |

Either set both Langfuse keys, or neither. Half-configured fails at startup.

---

## 🛡️ Privacy & offline guarantees

- 🚫 **No remote endpoints.** The Docling + PyMuPDF parser workers run under a `seccomp` filter that blocks every network syscall — even a malicious PDF can't phone home.
- 🪪 **No telemetry.** Memex doesn't ping anyone, ever. Logs stay on disk. Langfuse tracing is opt-in.
- 🗝️ **Vault locked down.** `~/.memex/vault` is mode 0700.
- 🔐 **MCP HTTP is auth-or-loopback.** The HTTP transport refuses to bind to anything except loopback unless a bearer token is set. No "oops, exposed the LLM to the LAN" footgun.
- 🪶 **Local LLM by default.** vLLM runs your chosen 7B/8B model on your GPU. No OpenAI / Anthropic / Mistral API keys, no third-party inference.
- 📡 **Air-gap test passes.** Pull the ethernet, do everything you'd normally do. Memex doesn't blink.
- 🗄️ **Vault backup is a `systemctl --user enable --now memex-vault-backup.timer` away.** See [`docs/deploy/backup.md`](docs/deploy/backup.md) — encrypted incremental snapshots via restic, default local repo at `~/.local/state/memex/backups/`, S3 / Backblaze / SSH targets documented for off-site.

The only thing that talks to the network is the *initial model download* (one-time HuggingFace pulls, gated through `scripts/download-models.py`). After that, you can pull the cable. 🔌

---

## 📂 What's in the vault

```
~/.memex/vault/
├── documents/
│   ├── 2f96ae1c-some-paper.md          ← canonical Markdown (the source of truth)
│   └── 2f96ae1c-some-paper/
│       └── source.pdf                  ← original ingested file
└── .memex/
    ├── embeddings.lance/               ← vector index (LanceDB)
    ├── search.sqlite                   ← keyword index (SQLite FTS5)
    ├── tables.sqlite                   ← structured table store for text-to-SQL (Table-RAG)
    ├── graph.ryu                       ← entity + citation graph (RyuGraph)
    ├── events.sqlite                   ← in-process audit bus (30-day prune)
    ├── manifests/{doc_id}.json         ← per-doc audit trail (sha, parse versions)
    ├── locks/{doc_id}.lock             ← fcntl advisory lock files (P1.5)
    └── daemon/vllm.{pid,log}           ← only used by `memex daemon start`;
                                          unused under systemd
```

**The Markdown files in `documents/` are the source of truth** 📜 (ADR-0003). Everything in `.memex/` is regenerable from them via `memex reindex`. You can git-version the canonical markdown, sync it across machines with Syncthing or rsync, edit by hand — the watcher catches the edit and rebuilds the indexes.

---

## 🧪 Run the tests

```sh
uv run pytest                  # ~1697 tests, ~30 seconds, no GPU needed
uv run pytest tests/unit       # just the pure-function tests
uv run pytest tests/integration  # full ingest→parse→index→ask flow with faked I/O
```

Integration tests fake the heavy I/O (vLLM, Docling, PyMuPDF worker, LanceDB, sentence-transformers) so the full pipeline can run on a laptop without a GPU. Five tests skip on Windows (seccomp + cross-process fcntl).

---

## 🧭 Documentation

| If you want to know… | Read… |
|---|---|
| 🌅 The why (vision + design principles) | [`docs/VISION.md`](docs/VISION.md) |
| 🔧 The how (engineering rules + stack) | [`docs/GUIDELINES.md`](docs/GUIDELINES.md) |
| 🗺️ What's done & what's queued | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| 🏗️ The architecture blueprint | [`docs/IMPLEMENTATION-PLAN.md`](docs/IMPLEMENTATION-PLAN.md) |
| 📐 Why we picked what we picked | [`docs/adr/`](docs/adr/) (ADRs 0001–0022) |
| 🚀 Network-facing MCP setup | [`docs/deploy/mcp-http.md`](docs/deploy/mcp-http.md) |
| 🖥️ systemd deployment (Linux) | [`docs/deploy/systemd.md`](docs/deploy/systemd.md) |
| 🍎 launchd deployment (macOS dev) | [`docs/deploy/launchd.md`](docs/deploy/launchd.md) |
| 🔬 Audit reports (E2E + load + OCR A/B + bug-hunt) | [`docs/audits/`](docs/audits/) |

Browse the same content as a navigable site:

```sh
uv sync --extra docs && uv run mkdocs serve
```

Material for MkDocs, dark palette, no Google Fonts (because of course not). 🌒

---

## 📜 License

Apache-2.0. Do as you wish. 🕊️

---

*Built around five principles: local-first by construction, Markdown as the source of truth, small models used well, observable at every layer, composable not captive. None of these are aspirational — every one is enforced by the test suite.* 🌱
