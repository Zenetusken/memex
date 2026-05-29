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
- 📖 **Read** your documents into clean Markdown (with figures, tables, equations preserved).
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

Memex ships **two hardware-tier profiles**. The 12 GB tier uses Qwen3-8B-AWQ and is the default; the 8 GB tier uses Qwen3-4B-AWQ at a tighter vLLM memory fraction. See [`docs/deploy/hardware-tiers.md`](docs/deploy/hardware-tiers.md) for the env-var matrix + the eval-verified quality numbers behind each profile.

---

## ⚡ Install

```sh
# 1. Clone + sync
git clone https://github.com/Zenetusken/memex.git ~/project/Doc_Flo
cd ~/project/Doc_Flo
uv sync --extra models --extra parse --extra serve

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

Each PDF goes through: validation → ingest copy → parse (PyMuPDF for born-digital, Docling for scans) → index (chunks → LanceDB + FTS5) → enrich (entity extraction + graph edges via vLLM). All inline, all logged.

For scanned content add `MEMEX_PARSE_DOCLING_OCR=1`; for born-digital PDFs (PowerPoint exports, LaTeX, Word) leave it off — the PyMuPDF pre-filter handles those at ~3× Docling's speed.

**Office documents** (`.pptx`/`.docx`/`.xlsx` + their ODF cousins) are first converted to PDF via headless LibreOffice, then run through the normal PDF pipeline — so a diagram-heavy slide deck's figures flow through the VLM diagram-transcription + chart-OCR passes (enable the VLM with `MEMEX_PARSE__DISABLE_VLM=false` for diagram-rich decks). The converted PDF is cached per document and reused on re-parse. LibreOffice must be installed (`soffice` on PATH). On the 12 GB tier `memex ingest`/`index`/`reindex` pause vLLM for the duration so the embedder isn't starved by a co-resident vLLM — no manual `MEMEX_RERANK_BATCH_SIZE` or daemon juggling needed. The VLM is **Qwen3-VL-8B-AWQ**, run as a short-lived vLLM process *during parse only* (the orchestrator is paused to free the GPU); it transcribes directed flow/state diagrams correctly where the older Qwen2.5-VL flattened them into a list. Tune via `MEMEX_MODELS__VLM_SERVING` (`vllm` | `transformers`) + `MEMEX_MODELS__VLM_SERVE__*` (port / `gpu_memory_utilization` / `max_model_len`) — see [`docs/specs/vlm-vllm-serving.md`](docs/specs/vlm-vllm-serving.md).

### Ask grounded questions

```sh
uv run memex ask "What does the paper say about ablations?"
uv run memex ask "Compare how Smith and Tan handle reflexivity"
uv run memex ask "Has anyone discussed CUDA-graph capture overhead?"
```

Output is JSON on a pipe, rich tables in a terminal. Every `claim` carries a `source_chunk_id`; the agent **refuses** when chunks don't ground the answer (returns `answered: false` + `refusal_reason`).

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
uv run memex mode set fast          # GPU reranker, 6 K context, ~14 s/ask
uv run memex mode set full          # reranker→CPU, 24 K context, slower rerank
```

`fast` is low-latency top-k RAG; `full` frees the GPU into a ~24 K orchestrator window (the reranker moves to CPU, ~20 s/query). `memex mode set` restarts the daemon-managed orchestrator; set `MEMEX_MODELS__CO_RESIDENCE_MODE` + restart `memex serve web` to also move the retrieval models. The full matrix + the eval numbers are in [`docs/deploy/hardware-tiers.md`](docs/deploy/hardware-tiers.md); the design is [ADR-0007](docs/adr/0007-co-residence-resource-modes.md).

### Browse the vault

```sh
# CLI listing
uv run memex list documents

# Or the web UI
open http://127.0.0.1:7423
```

The web UI gives you a documents list, **side-by-side preview of the source PDF** (server-rendered page images, lazy-loaded — works in every browser regardless of the "download PDFs" setting), a Cytoscape graph view of entity neighbors, and an inline edit-then-save flow (with conflict detection if someone else changed the doc since you started editing — see [`docs/deploy/mcp-http.md`](docs/deploy/mcp-http.md) for the 409 conflict surface). Long agent/summarizer runs surface **live progress** via an HTMX long-poll widget (no SSE / no new JS) — the per-section counter ticks on Summarize, the agent's node-by-node phase advances on Ask; the answer + summary panels label sources by **document title › section** (the raw `docid#hash` is kept as a tooltip), so a long deck's claims read as English rather than hashes.

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
```

Surfaces the documents most related to this one through the entity graph — ranked by the **specificity** of the entities they share, so a sibling that shares one rare concept beats one that shares five generic terms (a near-universal entity, or an incidental person/place name, is filtered out). Each result shows the connecting entities — the *why*. This is the discovery surface over the graph (ADR-0011); also a "Related documents" section on the web UI document view and the MCP `related_documents` tool. (It is *not* in the `/ask` retrieval path — a measured audit showed 1-hop graph expansion adds nothing to answering at this corpus scale, so it's a deliberate discovery feature, not passive recall-boosting.)

### Update to a newer Memex

```sh
uv run memex upgrade
```

Three steps in order: `git pull --ff-only` → `uv sync --extra models --extra parse --extra serve` → `systemctl --user restart` of every installed `memex-*.service`. Refuses if your tree has uncommitted changes (run `git stash` yourself if you want to keep them). vLLM's restart blocks ~30 s for the `Type=notify` readiness gate; the CLI prints a note so you don't mistake it for a hang.

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

Seven tools are exposed:
- 🔎 `search(query, k)` — hybrid retrieval over the vault
- ❓ `ask(question, scope_doc_ids?, scope_set?)` — full grounded answering agent
- 📝 `summarize(doc_id, instruction?, detail?)` — structured grounded document summary
- 📄 `get_document(doc_id)` — canonical markdown + frontmatter
- 📚 `list_documents()` — every doc in the vault
- 📌 `list_scope_sets()` — every saved document scope set
- 🌐 `get_graph_neighbors(doc_id)` — one-hop entity neighbors
- 🔗 `related_documents(doc_id)` — related docs ranked by shared-entity specificity (discovery)

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
| `MEMEX_VLLM_MODEL` | `Qwen/Qwen3-8B-AWQ` | HuggingFace model ID for the orchestrator. 8 GB tier: `Qwen/Qwen3-4B-AWQ`. |
| `MEMEX_VLLM_HOST` | `127.0.0.1` | Bind host. |
| `MEMEX_VLLM_PORT` | `8000` | Bind port. |
| `MEMEX_VLLM_QUANTIZATION` | `awq_marlin` | Quantization kernel. Set `""` for unquantized or FP8 models, `awq` for the legacy kernel. |
| `MEMEX_VLLM_MAX_MODEL_LEN` | `6144` | Max sequence length. Sized to fit the production answer prompt at `top_k=5` with chunks truncated to 1800 chars; the +2048 over the earlier 4096 ceiling costs ~1 GB KV-cache reservation under fp8_e5m2. |
| `MEMEX_VLLM_GPU_FRACTION` | `0.72` | 12 GB-rig floor with the 8B-AWQ orchestrator. **Drop to `0.68` when chart-OCR is enabled** (the default since 2026-05-23) — embedder + reranker + chart-OCR slot need the extra headroom. **8 GB tier: drop to `0.50`** to leave room for embedder + reranker alongside the smaller orchestrator. |
| `MEMEX_VLLM_KV_CACHE_DTYPE` | `fp8_e5m2` | Halves KV-cache memory for AWQ-int4 checkpoints. **Set to `auto` for FP8-checkpoint models** (vLLM blocks fp8 KV cache + FP8 weights at startup). |
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
uv run pytest                  # 913 tests, ~14 seconds, no GPU needed
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
| 📐 Why we picked what we picked | [`docs/adr/`](docs/adr/) (ADRs 0001–0009) |
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
