# ADR-0018: Companion-document merge — lecture transcript ↔ slide deck (grounded)

**Status:** Accepted (v1 shipped 2026-06-04, merge `f7fbba5`; keyframe-OCR amendment same merge)
**Extends:** [ADR-0017](0017-audio-asr-ingestion-route.md) §"Companion-document merge (deferred to Phase 2)"
**Spec:** [`docs/specs/companion-merge.md`](../specs/companion-merge.md)

## Context

A class lecture has two halves already in the vault as separate documents: the **slide deck** (a PDF)
and the **teacher's spoken commentary** (the ADR-0017 transcript). The transcript is ALREADY a
first-class grounded RAG document — same chunking/embedding/FTS/`verify`/citation path as any PDF
(validated live: a grounded ask cited transcript chunks at `confidence:high`). The user's requirement
is that the two be tied as **jointly grounded factual evidence** — "the teacher's lectures and these
documents are considered grounded factual information" — **not** a navigation cross-link bolted on top.

ADR-0017 deferred the merge to Phase 2 and v1 carried the hooks (per-segment GLOBAL `start_s` + char-
spans + `Chunk.time_range`). Both the CR350 video transcripts AND their `Cours N.pdf` / `Semaine N.pdf`
slide decks are now indexed, so the merge is buildable on real data.

A pre-design measurement (2026-06-03) settled an adjacent question — whether transcript chunks need a
relevance/salience classification layer to keep noise out of retrieval. **Measured NO:** the worst-case
(intro) lecture is only 2.7% true noise (social/artifact), 29.5% is legitimately-queried *admin*
content, and dense retrieval leaks ZERO noise into real-query top-10. The embedder's semantic relevance
is already the right "layer." So the merge builds on clean, already-grounded transcripts.

## Decision

Build a **user-paired** companion-merge that aligns each transcript CHUNK to its best slide-deck PAGE
by **EmbeddingGemma cosine** (the MaViLS method), stores the alignment as **derived sidecar state**,
writes a **document-level CITES link**, and — the load-bearing part — makes slide + commentary
**jointly groundable** via a **HARD-gate-safe retrieval AUGMENTATION**, not chunk fusion.

Load-bearing decisions:

1. **Grounded integration via RETRIEVAL AUGMENTATION, not chunk fusion.** When the retriever surfaces
   one side of an aligned pair, a default-OFF `augment_companion` node pulls the aligned counterpart
   chunk(s) into the candidate pool (a deck page → the transcript chunks that explain it, and
   vice-versa). Each chunk stays PURE (own content-addressed `chunk_id`, own source); `verify` grounds
   every claim against its OWN chunk. **HARD-gate-safe by the `expand_graph` construction** — it only
   adds REAL indexed chunks, never fabricates/fuses/mutates, so no hallucination path opens and a
   counterfactual stays refused. **Chunk-fusion (slide+commentary in one chunk) is REJECTED** — it
   churns `chunk_id`s and injects mixed-source text into grounding (a real gate risk).

2. **The alignment is DERIVED + navigation-grade, never on the grounding path itself.** The per-chunk→
   page map is a regenerable sidecar (`vault/.memex/companion_alignments.json`, atomic write); the
   doc-level link is a reciprocal CITES edge (`related_documents` surfaces it). Neither is read by
   `answer_query`/`verify`/the `chunk_id`/the embeddings — the `Chunk.page` + `related_documents` +
   `scope_sets` precedents. Only the §1 augmentation touches retrieval, and it is gated + eval-gated.

3. **Honest accuracy: transcript-text→slide-text cosine is MaViLS's WEAKEST modality (F1 ≈ 0.53),** not
   its multimodal 0.82 — we discard the strong video-FRAME-OCR signal (0.76) per ADR-0017's "visual
   track ignored." v1 is per-chunk argmax cosine + a null floor (`τ_null`, honestly un-calibrated) + a
   cheap monotonic tie-break. The MaViLS asymmetric-jump DP (with our REAL `start_s` time prior) and
   video-keyframe-OCR matching (recovers the discarded strong signal) are scoped FAST-FOLLOWS.

4. **Pairing is USER-SPECIFIED** (`memex link-slides <transcript> <deck>`). Auto-pairing is non-trivial
   (Cours-04 video ↔ *Semaine*-4 deck) and a wrong pair mis-attributes commentary — deferred to a
   SUGGEST-only layer.

## Consequences

**Positive:** slide + the teacher's explanation answer questions together, each grounded to its real
source; the transcript stays first-class; navigation between a `[mm:ss]` point and its slide; reuses
the embedder + graph + sidecar precedents (no new model/runtime).

**Negative / trade-offs:** transcript-only alignment is modest (~0.53 F1) until the DP / keyframe
fast-follows; `τ_null` is un-calibrated without a gold set; the augmentation is HARD-gate-ADJACENT so
it ships default-OFF behind an eval gate; user must name each pair.

**HARD-gate posture:** alignment + link + nav surfaces are neutral by construction; the augmentation is
validated (`refusal_cf=1.0` must hold with it ON over a real pair) before any default-ON flip.

## Alternatives considered

- **Chunk-fusion (one slide+commentary chunk)** — REJECTED: chunk_id churn + mixed-source grounding.
- **Nav-only sidecar (no grounding effect)** — INSUFFICIENT: the user explicitly wants grounded
  integration, "not just logically integrated." Kept as the always-safe fallback if the augmentation
  eval doesn't clear.
- **A transcript salience/classification layer** — MEASURED-AWAY (2.7% noise, 0 retrieval leakage; the
  embedder already filters; admin ≠ noise).
- **Auto-pairing by name/dir** — REJECTED for v1 (silent mis-pairing risk).
- **A new alignment model / learned fusion** — REJECTED: reuse EmbeddingGemma; MaViLS found no fusion
  gain.

## Revisit when

- v1's argmax under-performs on a real pair → ship the asymmetric-jump DP + the `start_s` prior.
- ~~Alignment is still weak → video-keyframe-OCR matching~~ → **DONE (Amendment 2026-06-04).**
- A gold transcript↔slide alignment set exists → calibrate `τ_null` + the (query,doc)/(doc,doc) A/B.
- The augmentation eval clears with a measured win → consider default-ON.

## Amendment (2026-06-04): video keyframe-OCR matching SHIPPED

The principled accuracy lever from decision 3 / "Revisit when" is now implemented (opt-in,
video-gated) — spec `docs/specs/companion-merge.md` §14. For a transcript whose source is a VIDEO
(a screen-recorded lecture showing the slides), each chunk's slide is taken from the on-screen FRAME
at its `time_range` midpoint (PyAV decode → Qwen3-VL OCR → cosine to the deck), recovering MaViLS's
strong frame-text modality (F1 ≈ 0.76) over the transcript-text cosine (F1 ≈ 0.53). Keyframe-PRIMARY
above the floor, transcript-text FALLBACK below it. `parse/keyframe_ocr` owns frames+OCR+cache;
`index/companion.compute_alignment(keyframe_texts=)` does the embed+match (the `index/` ↛ `parse/`
boundary holds — the CLI wires them); `link-slides create --use-video`. Content-addressed OCR cache
keeps the derived sidecar reproducible. HARD-gate-neutral (still a derived sidecar; the augment node is
untouched).

**Validated** on Cours 03 ↔ Cours 3 against 18 hand-labeled gold frames (argmax-correctness, demo +
off-deck frames included), via a floor SWEEP. **At the calibrated/shipped floor 0.80** the `--use-video`
system (keyframe-primary + transcript fallback) scores **79% on-slide vs 50% transcript-only (+29%)**,
off-slide fallback **4/4** (the old 0.50 floor was 71%/+21% with 1/4 fallback — the floor buys the
off-slide robustness). `companion_keyframe_min_score = 0.80`: true matches cluster ≥0.82, demo/off-deck
false matches 0.64–0.78 (cleanly dropped); the separation isn't perfect — one on-slide error survives
(a 0.85 frame matching an ADJACENT lookup-step slide, same topic). A too-high floor only falls back more
(conservative), never forces a wrong slide. **Calibrated on ONE deck — re-check before treating 0.80 as
universal.** Caveat: the deck's `Chunk.page` is navigation-grade and can drift from the true PDF page on
a figure-heavy deck, so the alignment matches by CONTENT (`deck_chunk_id`) and the scoring mapped each
predicted chunk to its true PDF page by text-overlap; a citation-grade page map is a separate follow-up.

## Amendment (2026-06-05): the three deferred §13 levers — resolutions

Closing the "deferred, each its own arc" list (spec §13). **Shipped + merged to main 2026-06-05** (via
`feat/companion-levers`, 13 commits — DP `70fca71`/`5adbaae`, dedup-revert `8f0ad57`→`aade5d7`, page-map
`0340eb9`/`6b6bfd9`/`52fb439`, webui page-jump fix `15264ce`). All three stay HARD-gate-neutral (the
alignment is the default-off augment sidecar; keyframe OCR is parse-stage; `Chunk.page` is nav metadata).

- **MaViLS asymmetric-jump DP + real `start_s` prior — SHIPPED OPT-IN.** `align_blocks(use_dp=True)` → a
  pure Viterbi (`_align_dp`): emission = cosine; transition = `-companion_dp_lambda_jump` × jump (forward
  1× / backward 2× / stay 0); a `companion_dp_time_weight` `start_s` prior; keyframe-PRIMARY chunks are
  FIXED anchors; null carries the page context. **Default OFF** (`companion_align_dp_enabled`, off ⇒
  byte-identical to the greedy tie-break). The DP logic is unit-validated; the **corpus win is
  UNMEASURED** — enabling + tuning `λ_jump` awaits a transcript→slide gold set (the existing 18-frame
  gold is keyframe→slide). So it ships as a ready, off-by-default lever, not a measured improvement.
- **Perceptual-hash keyframe-OCR dedup — REJECTED (built then reverted) as fundamentally unviable.** A
  whole-frame perceptual hash (aHash or dHash, any resolution) cannot separate a DIFFERENT slide that
  differs by small text (an animation build / one-word bullet → ~1–2 bits) from the SAME slide HELD under
  a moving webcam/demo overlay (~17–48 bits) — the dangerous case is *closer* than the safe one, so no
  Hamming threshold works, and the 0.80 keyframe floor does NOT contain a false dedup (slide A's reused
  text legitimately matches deck page A above the floor → a confidently-wrong page). The only safe dedup
  compares OCR'd TEXT (needs OCR first, no savings), and the keyframe OCR is already cached (one-time
  cost). Don't retry a whole-frame-hash dedup for slide frames.
- **Citation-grade deck page-map — WIRED + real-deck validated.** The `Chunk.page` drift is real but
  PRESENTATION-ONLY (retrieval is content-addressed via `deck_chunk_id`). The fix is a transient
  page-boundary marker that rides the body transforms as a ruler then is stripped — measuring per-page
  spans against the exact chunked body while keeping it byte-identical (zero chunk_id churn). The
  foundation (`core/text.py` marker helpers + the `collapse_consecutive_duplicates` exclusion + the golden
  byte-stability invariants through `_finalize_body` AND reattach+linearize) is now threaded end-to-end:
  - **Parse** (`_finalize_body_with_page_starts`, all 3 page routes): records each page's true `char_start`
    in `doc.body` via the marker round-trip. The body returned is ALWAYS `_finalize_body(plain_markdown)`,
    so `doc.body` + every chunk_id stay byte-identical BY CONSTRUCTION; char_starts are recorded only when
    the round-trip reproduces the canonical body byte-for-byte, else the doc stays nav-grade.
  - **Manifest**: `PageDecision.char_start` (default `-1` = legacy/unknown). Pydantic-optional → old
    manifests load unchanged; a re-parse upgrades a doc.
  - **Index** (`_exact_page_intervals`): maps the parse-recorded boundaries through the SAME chart-reattach
    + GFM-linearize transforms to citation-grade intervals in the indexed body, byte-equality-GUARDED →
    nav-grade `page_char_counts` fallback on any mismatch. `_citation_grade_boundaries` gates on CONTENT
    pages so a figure-only slide (`char_count 0 → char_start -1`) never demotes the whole deck.
  - **Chunker**: `exact_page_intervals` takes precedence over the nav-grade `char_count`-derived intervals.

  **Activation** is route-dependent and fail-safe: PyMuPDF + scan/image build `conversion.markdown` AS the
  per-page join → always reconstruct; the Docling document-level markdown is a whole-doc
  `export_markdown_header_aware` serialization that equals the join after VLM escalation (re-stitched as
  the join — the figure-heavy-deck target) or on a deck with no separator divergence, else it
  fail-safes to nav-grade. The guards mean a wiring bug can only LOSE page precision, never churn chunk_ids.
  **Validated on a real 24-page figure-heavy Docling deck (cours-12-ipv6):** citation-grade activated
  24/24 pages; the index guard passed through 4 chart re-attaches + 4 table linearizations (body
  15520→22237); chunk_ids byte-identical exact-vs-nav; 51/62 chunks corrected (nav-grade mis-attributed by
  ~6.7KB of chart-shift), p1=title / p2=agenda matching the true slides. Pinned by `test_page_map_wiring.py`
  (both pure helpers + the figure-only-page gate), `test_page_markers.py` (the index-transform golden), and
  parse→index→`Chunk.page` integration tests on the PyMuPDF, Docling (activation + fail-safe), and scan
  routes. **Migration**: a doc reaches citation-grade on its next re-parse+reindex — and an escalated deck
  MUST re-parse with VLM enabled (`MEMEX_PARSE__DISABLE_VLM=false`, cache-replays byte-identical), since
  `memex parse`'s default `disable_vlm=True` would otherwise drop the VLM transcriptions; a documented
  follow-on like every other re-parse migration, not auto-applied.
