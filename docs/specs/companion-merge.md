# Spec: companion-document merge — lecture transcript ↔ slide deck (grounded)

**Status:** **Proposed** (design — NOT built). **ADR:** [ADR-0018](../adr/0018-companion-document-merge.md)
(amends/extends [ADR-0017](../adr/0017-audio-asr-ingestion-route.md) §"Companion-document merge").
**Research basis:** MaViLS (arXiv:2409.16765) + the 4-arm codebase/method research (2026-06-03).
**Precedents reused:** `Chunk.page` navigation attribution; `expand_graph` candidate augmentation;
`related_documents`/CITES discovery; `core/scope_sets.py` atomic-JSON sidecar; the EmbeddingGemma
native-prompt seam.

> A class lecture has two halves: the **slide deck** (the document) and the **teacher's spoken
> commentary** (the transcript, ADR-0017). This spec aligns them and makes them **jointly grounded
> factual evidence** in the RAG — not a navigation cross-link bolted on top.

**Scope note (this is a DESIGN — none of the below is built yet).** The codebase SEAMS this spec
reuses all exist and are verified (`page_intervals`, `FTSStore.chunks_for_document`, `hybrid_search`/
`_embed_query`, `embed_prompts.document_input`, `registry.use("embedder")`, `complete_structured`,
`GraphStore.link_cites`/`.citations`/`.related_documents`, the `core/scope_sets.py` atomic-write
pattern, the `expand_graph` lazy-store/default-OFF node contract, `Chunk.time_range`/`Chunk.page`). The
NEW code the spec PROPOSES — to be built across the increments below — is: (i) `AgentsSettings`
companion fields (§11); (ii) an `index/companion.py` alignment+store module (§3, §5); (iii) the CLI
`memex link-slides` verb (§4); (iv) the `augment_companion` `/ask` node (§7); (v) the surfaces (§8).
**Build order (each its own per-cycle-validated increment, the audio-feature discipline):** B1 pure
alignment core (page/transcript chunk read → embed → cosine → argmax+null+tie-break, unit-tested faked-
embedder) → B2 sidecar store + `link_cites` doc-link + the `link-slides` CLI → B3 the surfaces (CITES-
driven doc-view "Companion" line + answer companion chip) → B4 the eval-gated `augment_companion` node
(default-OFF; the §9 HARD-gate eval before any default-ON flip). B1–B3 are HARD-gate-neutral by
construction; only B4 is gate-adjacent.

## 1. Requirement #1 (the load-bearing one) — GROUNDED integration, not "logical" linking

**The transcript is ALREADY a first-class grounded RAG document** — the ADR-0017 ingestion chunks it,
embeds it into the SAME vector store, indexes it in the SAME FTS, and the `/ask` graph retrieves +
`verify`-grounds + cites it identically to a PDF — by construction (no transcript-specific grounding
path exists; `Chunk.time_range` is navigation-grade metadata the grounding path never reads). This was
confirmed by a live grounded ask during the audio-ingestion arc (the CR350 exam-policy question
grounded + cited transcript chunks from Cours 01 AND the Cours 05 video at `confidence:high` — a live
run, see [[audio-video-asr-shipped-2026-06-03]]; not a committed regression test). The teacher's words
are grounded factual evidence **today**. This spec must not demote that.

The merge's job is to make the **slide and the teacher's explanation of it a single body of grounded
fact**, so a question about a slide's topic can be answered from the slide AND the spoken commentary
**together**, each cited to its own real source. Three properties, in priority order:

1. **G1 — Joint grounding (the requirement).** When the retriever surfaces a chunk from one side of an
   aligned pair, its aligned counterpart is available as grounded evidence too, so the answer draws on
   both. **Realized as a retrieval AUGMENTATION (additive, per-chunk-pure), NOT chunk fusion** —
   *per-chunk-pure* meaning every chunk is left BYTE-UNMODIFIED (its content-addressed `chunk_id`, text,
   and single source document are untouched); the merge only changes WHICH chunks are candidates, never
   what a chunk IS.
2. **G2 — Structural tie.** A persisted, navigable transcript↔deck relationship (document-level link +
   per-chunk→slide alignment) so the two are explicitly companions.
3. **G3 — Navigation.** From a `[mm:ss]` transcript point → its slide; from a slide → the spoken
   commentary on it.

**The HARD gate is preserved by construction** (G1 adds only REAL chunks to the candidate pool; `verify`
still grounds every claim against its own source chunk — no fabrication path; §9).

## 2. What the merge aligns (the data, all already on disk)

- **Transcript side** — the indexed CHUNKS (the retrieval/display unit), each carrying
  `Chunk.time_range: tuple[float,float]|None` (GLOBAL seconds, ADR-0017) + `chunk.text`. Read via
  `FTSStore.chunks_for_document(transcript_doc)`. (Aligning at the chunk level — not the finer
  manifest `TranscriptSegment` — keeps the alignment in the SAME unit the surfaces + the G1
  augmentation key on; segment-level is a deferred refinement, §13.)
- **Deck side** — the indexed CHUNKS of the slide-deck doc, each carrying `Chunk.page` (the slide
  number, attributed from `ParseStage.pages[].char_count`, navigation-grade). Read via
  `FTSStore.chunks_for_document(deck_doc)`. **We embed each deck chunk's INDEXED `chunk.text`** (the
  post-finalize content the retriever already serves) and read its already-computed `Chunk.page` — we
  do NOT re-slice the deck body via `page_intervals`, so the post-stitch char-count drift (chart-OCR /
  `[table-rows]` shifting body offsets — the R1 / `chunker.py` navigation-grade caveat) never touches
  the alignment text. The residual imprecision is only the `Chunk.page` attribution's own ±1
  navigation-grade slip, which is acceptable for an approximate slide target. (A deck parsed with
  all-zero per-page `char_count`s — a legacy/pre-2026-05-27 manifest — has `Chunk.page=None`; the slide
  decks must be FRESHLY parsed, else `link-slides` errors with a clear "deck has no page attribution;
  re-parse" message rather than producing a pageless alignment.)
- **Embeddings** — both sides re-embedded on demand via the live EmbeddingGemma (`registry.use(
  "embedder")`), NORMALIZED (so cosine == dot). No new model, no training. (LanceDB exposes no raw
  stored-vector export, so re-embed; the merge is an offline op — cost is acceptable. A
  stored-embedding fetch is a deferred optimization, §13.)

## 3. The alignment algorithm (MaViLS, honestly calibrated)

**v1 = per-transcript-chunk argmax cosine with a null floor + a cheap monotonic tie-break.** The full
MaViLS asymmetric-jump DP is a scoped **fast-follow** (§13), not v1.

1. **Embed the deck pages.** For each deck chunk, embed `chunk.text` **document-side**
   (`embed_prompts.document_input(chunk_title(c), c.text)` = `"title: … | text: …"`). One vector per
   deck chunk; group by `chunk.page` (a page may have >1 chunk → keep all, the page's vectors).
2. **Embed the transcript chunks.** For each transcript chunk, embed its text **query-side**
   (`retrieve/hybrid._embed_query` → `"task: search result | query: …"`). **v1 COMMITS to the
   (query=transcript, document=deck) asymmetry** — rationale: "which slide ANSWERS this narration" ≈
   query→document, EmbeddingGemma's trained retrieval orientation. The (query,doc)-vs-(doc,doc)
   comparison is a **one-time BUILD-TIME calibration during implementation** (run both on the CR350
   pair, eyeball which alignment is better, hardcode the winner) — **NOT a runtime config knob** (no
   `companion_embed_*` setting; the chosen orientation is fixed in the alignment module).
3. **Similarity** `S[i,j] = T̂[i] · P̂[j]` (both L2-normalized → one numpy `T @ P.T`), `S ∈ [-1,1]`.
4. **Assign.** For transcript chunk `i`: `page(i) = page-of(argmax_j S[i,j])`, **BUT** if
   `max_j S[i,j] < τ_null` → **NULL** (an off-slide tangent / no good slide — the teacher digressing).
   `τ_null` is the config knob `AgentsSettings.companion_align_min_score` (default `0.40`),
   **honestly labelled UN-CALIBRATED** — there is no gold alignment set yet, so `0.40` is a starting
   guess to be eyeballed on the CR350 pair, not a validated threshold (§12); it is exposed so a user
   can tune it per-corpus.
5. **Monotonic tie-break (cheap, not full DP):** walking chunks in time order, track the last assigned
   non-null page `p_prev`; when chunk `i`'s top-2 scores are within `ε` (small, e.g. `0.02`) AND one of
   the tied pages is `≥ p_prev`, pick the smallest such page `≥ p_prev` (lectures advance). It NEVER
   overrides a clear argmax (only breaks near-ties) and NEVER forbids a backward jump (a clear backward
   argmax wins) — the asymmetric-penalty DP (§13) is the principled version.

**Honest accuracy expectation:** transcript-text→slide-text cosine is MaViLS's *weakest* single
modality, **F1 ≈ 0.53** (we discard the strong video-FRAME-OCR signal, F1 0.76, per the ADR-0017
"visual track ignored" amendment). The spec does NOT anchor on MaViLS's multimodal 0.82. Our genuine
edge over MaViLS is that the stored GLOBAL `start_s` gives a REAL time→slide prior (vs their estimated
one) — used as a candidate-window narrowing in the DP fast-follow, not v1. **The biggest accuracy
lever, if v1 under-performs, is sampling video KEYFRAMES + matching frame-OCR to slides — recovering
the discarded 0.76 signal; that is the principled upgrade, not transcript-cosine tuning (§13).**

The alignment is **PURE + deterministic** given fixed embeddings → reproducible; it carries the
`embedding_recipe_version` that produced it, and re-runs if that bumps (the `Chunk.page` / chart-cache
discipline).

## 4. Pairing — USER-SPECIFIED in v1

Auto-pairing is out of scope (the real pairing is non-trivial: `Cours-04` video ↔ `Semaine-4` deck;
a wrong pair would mis-attribute commentary). v1 is an **explicit CLI verb:**

```
memex link-slides create <transcript_doc_id> <deck_doc_id>   # align + persist + CITES-link
memex link-slides list                                       # all pairs (aligned/null counts)
memex link-slides delete <transcript_doc_id> <deck_doc_id>   # drop a pair's sidecar entry
```

`create` (a) computes + stores the alignment (§5), (b) writes the document-level link (§6). It is
**IDEMPOTENT — re-running re-embeds + re-aligns + REPLACES the pair, so re-running IS the refresh** (no
separate `--refresh` flag; `compute_alignment` always recomputes from the live chunks + embeddings). A
future inference layer (course-code/ordinal, the `course_refs` resolver precedent) may SUGGEST pairs,
but the link is only written on the user's explicit command.

## 5. Storage — a derived-state sidecar

`vault/.memex/companion_alignments.json` (one file, atomic write mirroring
`core/scope_sets.py`: `mkstemp`→`fsync`→`os.replace`):

```json
{ "pairs": [ {
    "transcript_doc": "...", "deck_doc": "...",
    "embedding_recipe_version": "v1-gemma-prompts", "aligned_at": "<stamp>",
    "blocks": [ { "transcript_chunk_id": "...#h", "time_range": [t0, t1],
                  "deck_chunk_id": "...#g", "deck_page": 12, "score": 0.57 } ],
    "null_count": 3
} ] }
```

- **Keys:** `transcript_chunk_id` + `deck_chunk_id` are the PRIMARY identity (deterministic, content-
  addressed) — the G1 augmentation (§7) resolves a reranked chunk by `chunk_id`, so the chunk_ids are
  what it joins on. `deck_page` is a CACHED navigation hint (the deck chunk's `Chunk.page`, re-derivable
  from the chunk) for the §8 webui label — NOT the join key. `time_range` is a cached copy of the
  transcript chunk's anchor for the §8 label. So a re-index that churns chunk_ids makes BOTH ids stale →
  `null` resolutions in the augmentation (which silently skips them, fail-open) until a re-`create`. JSON
  is written with sorted keys for stable diffs.
- **DERIVED state** (regenerable from the indexed chunks + embedder) ⇒ in the `reindex_vault(force=
  True)` teardown allow-list (unlike user-authored `scope_sets`). A stale entry (chunk_ids changed) is
  re-derivable; re-running `link-slides create` rebuilds it (and `reindex --force` drops it).
- Read fails **OPEN** to "no alignment" — a corrupt/missing file makes the §7 `augment_companion` node
  **silently skip** (it is purely additive; `/ask` proceeds exactly as if no pair were aligned) and a
  doc view shows no companion chip; NEVER an error on the read path (the `resolve_scope_set_doc_ids`
  fail-open precedent). Only the management surface (`memex link-slides list`) surfaces a
  `VaultIntegrityError` loudly.
- The reverse index (deck_chunk_id/page → transcript_chunk_ids) is computed in-memory from `blocks` on
  load (no second store) — it powers the deck→commentary direction of §7/§8.

## 6. The document-level link — a CITES edge + reciprocal wikilink

Reuse the existing graph + wikilink machinery (NOT a new edge type): at `link-slides`, write a
**reciprocal `CITES` edge** via `GraphStore.link_cites(from_doc_id, to_doc_id, surface_text=
"companion lecture"/"companion deck", confidence=1.0)` (the EXACT live signature — verified
`index/graph_store.py:572`; NB it's `link_cites`, not `add_citation`/`upsert_citation`) in BOTH
directions, so `related_documents` / `GraphStore.citations` surface the companion on BOTH doc views
(the read-only discovery seam, HARD-gate-neutral). The per-slide granularity lives in the §5 sidecar,
NOT on the edge (the CITES schema is frozen — `surface_text`/`confidence` only;
`index/schemas/graph.cypher:43-47`).

## 7. G1 — the GROUNDED retrieval augmentation (the requirement, HARD-gate-safe)

**This is the part that makes the merge "grounded RAG, not just logical linking."** A new
**default-OFF**, kill-switched node `augment_companion` in the `/ask` graph, placed **immediately
AFTER `rerank`** (between `rerank` and `query_tables`). **NB the placement DIFFERS from `expand_graph`
deliberately:** `expand_graph` sits BEFORE `rerank` (`resolve_artifact_scope → expand_graph → rerank`,
`answering.py:2286-2287`) and broadens the unranked candidate pool; `augment_companion` sits AFTER
`rerank` because it is **targeted on the reranked WINNERS** — it only pulls the aligned counterparts of
the chunks that already won reranking, not of all 50 candidates (which would flood the pool with
loosely-aligned noise). It **mirrors `expand_graph`'s CONTRACT** — the lazy-store-open pattern
(`agents/ → index/` edge), the default-OFF kill-switch, and the additive-candidate-only invariant —
NOT its graph position.

- **`augment_companion`** — for the **top reranked chunks** (`state.reranked`), if a chunk belongs to a
  doc with a companion alignment (§5 sidecar), pull its **aligned counterpart chunk(s)** and APPEND
  them to `state.reranked`: a winning DECK chunk on page P → the transcript chunks aligned to P (the
  teacher's words on that slide); a winning TRANSCRIPT chunk → the deck chunks on its aligned page (the
  slide it explains). Bounded to at most `companion_augment_max` **TOTAL** counterpart additions per
  query (default `3`), collected greedily in rerank order across the winners and de-duped against the
  existing reranked set. The `score ≥ companion_align_min_score` floor is ALREADY enforced at
  alignment time — a below-floor block has `deck_chunk_id=None`/`deck_page=None` (a NULL block), so the
  counterpart lookup (`_companion_counterpart_ids`) skips it by construction. The appended counterparts are
  contextually relevant BY ALIGNMENT (they're the companion of a reranked winner), so appending them
  unranked is sound; assess/answer/verify then treat them like any other chunk.
- **HARD-gate-safe BY CONSTRUCTION** (the `expand_graph` argument): it only **ADDS real, already-
  indexed chunks** — never fabricates, never fuses, never mutates a chunk (chunk_ids untouched).
  `verify` grounds each claim against the chunk **its own `source_chunk_id` names** — there is NO
  cross-chunk grounding, so an added commentary chunk can only ground a claim THAT CITES IT (a claim
  about the commentary's REAL content), never "lend" its evidence to a false claim about the slide. So
  no hallucination path opens. A counterfactual stays refused (neither the slide nor its commentary
  contains the false premise → no grounding evidence → `refusal_cf` preserved — **a claim this spec
  ASSERTS but §9 EVAL-VALIDATES before default-ON, not by fiat**). It can only turn a refuse→answer
  when the added evidence is REAL (a correct, grounded answer — the desired behaviour).
- **Gating:** `AgentsSettings.companion_augment_enabled` (default **OFF** until the §9 HARD-gate eval
  validates it ON), `MEMEX_AGENTS__COMPANION_AUGMENT_ENABLED`. The node short-circuits when no pair is
  aligned (the vast majority of vaults). Like `expand_graph`, it's ANDed with a per-call param.
- **Why this and not chunk-fusion:** fusing slide+commentary into one chunk would churn the content-
  addressed `chunk_id` AND inject mixed-source text into grounding (a real gate risk, R3) — explicitly
  REJECTED (§13). The augmentation keeps every chunk PURE.

**Validation gate before default-ON:** the augmentation is HARD-gate-ADJACENT (it changes the candidate
pool), so it ships **default-OFF** and only flips ON after the §9 eval shows `refusal_cf=1.0` holds
with it enabled on the counterfactual corpora + a measured recall/answer win on a companion pair. If
the standalone cross-doc retrieval already grounds slide+commentary jointly (§9 measures this), G1 may
stay an opt-in enhancement rather than a default — **data decides.**

## 8. Surfaces (G2/G3 — read-only, HARD-gate-neutral)

- **Answer panel** (`_answer.html`): when a cited chunk has an aligned counterpart, a quiet "↔ slide
  12" (for a transcript cite) / "↔ lecture 24:01" (for a deck cite) companion chip next to the source
  link (reuses the `_source_view` view-model; the time chip already ships). When G1 added the
  counterpart and it was cited, both simply appear as sources.
- **Doc view** (`document.html`): a "Companion" line (the CITES edge → `related_documents` already
  surfaces it) — "Slides: CR350 - Cours 1" on the transcript, "Lecture: CR350 ZOOM Cours 01" on the
  deck. A per-slide "spoken commentary `[mm:ss]–[mm:ss]`" backlink (from the §5 reverse index) is a
  nice-to-have v1 surface.
- **CLI/MCP:** `memex link-slides list` shows pairs; the alignment is inspectable.

## 9. HARD-gate posture + the validation plan

- **G2/G3 (alignment + link + nav) are HARD-gate-neutral BY CONSTRUCTION** — a derived sidecar + a
  read-only CITES edge + presentation chips, none of which the `answer_query` graph / `verify` /
  chunk_ids / embeddings ever read for grounding (the `Chunk.page` + `related_documents` + `scope_sets`
  precedents). No eval needed for these.
- **G1 (augmentation) is HARD-gate-ADJACENT → eval-validated before default-ON:** run the standard
  answer-eval (`refusal_cf=1.0` / 0-hallucination) on the counterfactual corpora WITH
  `companion_augment_enabled=true` over a real aligned pair (CR350 Cours-01 ↔ Cours 1.pdf); it must
  HOLD. ALSO measure the win: a question whose answer needs BOTH the slide and the commentary — does
  augmentation lift answeredness/recall vs OFF? **First, measure whether plain cross-doc retrieval
  already grounds both** (it cited 2 transcripts cross-doc live) — if the joint grounding is already
  there, G1 is a recall safety-net, not the primary mechanism.
- **The transcript stays a first-class grounded doc throughout** — the merge adds, never subtracts.

## 10. Reproducibility & lifecycle

Deterministic given fixed embeddings; carries `embedding_recipe_version`; re-runs on a recipe bump or a
re-`create`; in the `reindex --force` teardown (regenerable — alongside `asr_cache.sqlite` etc.).
`link-slides create` pauses the orchestrator (`pause_vllm_for_gpu`) to free the GPU for the embedder
(mirrors ingest/index), then restarts it; the read-only `list`/`delete` touch no model.

## 11. Config / surfaces summary

- `AgentsSettings.companion_augment_enabled: bool = False`, `companion_augment_max: int = 3`,
  `companion_align_min_score: float = 0.40`.
- CLI `memex link-slides create <transcript> <deck>` (idempotent) + `list` + `delete`.
- Sidecar `vault/.memex/companion_alignments.json`; CITES edges; webui companion chips + doc-view line.

## 12. Eval / calibration

- **Alignment quality** — a small hand-checked gold (a few CR350 transcript-chunk→slide labels) to
  calibrate `τ_null` + the (query,doc) vs (doc,doc) A/B; report a coarse accuracy, honestly framed
  against the ~0.53 transcript-only floor (NOT 0.82). Clips/labels stay LOCAL.
- **G1 HARD-gate** — §9: `refusal_cf=1.0` holds with augmentation ON; a measured answer/recall win.

## 13. Out of scope (deferred, each its own arc)

- **The MaViLS asymmetric-jump DP** (`λ_jump≈0.1`, backward 2×/forward 1×/stay 0; lower for
  revisit-heavy lectures) + the **real `start_s` time→slide prior** (our edge over MaViLS) — the
  principled monotonic refinement once v1's argmax is measured.
- **Video KEYFRAME-OCR matching** — recovers MaViLS's discarded strong signal (F1 0.76); THE accuracy
  upgrade if transcript-only under-performs. Needs the video frames (we keep `source.mp4`).
- **Chunk-fusion** (slide+commentary in one chunk) — REJECTED for grounding safety (chunk_id churn +
  mixed source); a faithfulness-gated "contextualized" view is a separate, gate-sensitive arc.
- **Auto-pairing** (course-code/ordinal inference) — SUGGEST-only later; v1 is explicit.
- **Segment-level alignment** (finer than chunks) + **stored-embedding reuse** (vs re-embed) +
  **title/agenda-slide downweighting** (a slide that matches everything) + **principled `τ_null`
  calibration** (needs a gold set).
