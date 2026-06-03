# ADR-0017: Audio Ingestion via a Local ASR Route (Pluggable Backend, In-Process by Default)

- **Status**: Proposed
- **Date**: 2026-06-03
- **Deciders**: Memex core team
- **Tags**: parse, ingest, models, asr, audio, architecture

## Context

Memex ingests documents (PDF / Office / images / scans) but has **no audio path**. The user's
motivating use case is **class-lecture audio** — recorded MP3s, often 1–2 hours, **multilingual
and heavily French**, sometimes code-switching into English technical jargon (`VLAN`, `OSPF`,
`BGP`). Two shapes:

1. **Standalone** — a recorded lecture becomes a searchable, answerable transcript document.
2. **Companion** (a Phase-2 follow-up, see below) — a class *video/recording* whose audio
   **verbally explains a slide deck/PDF**; the spoken commentary *adds to* the slide text and
   must eventually be **aligned/merged** with its companion document.

This was already scoped at ROADMAP Tier-1 ("🎙️ Audio ingestion, USER USE CASE, scoped
2026-06-01"). The orchestrator (`Qwen3.5-4B`, ADR-0015) and the doc-VLM (`Qwen3-VL-8B`) are
**vision-language only** — neither has an `audio_config`, so audio needs a **dedicated ASR
model** ([[qwen-migration-research-2026-05-26]]). The right shape is **a new parse route,
analogous to scan→VLM** (`docs/specs/scan-vlm-parse.md`): detect an audio source → transcribe to
**timestamped Markdown** → hand to the **existing** chunk/embed/retrieve/answer pipeline, so a
transcribed lecture is searchable + answerable like any document, with the grounded HARD gate
intact (it grounds on the transcript text).

A pre-research document (an earlier team's "Whisper vs Qwen3-ASR on a 12 GB GPU" brief) was
adversarially re-verified against primary sources on 2026-06-03 (see
[[asr-audio-scope-2026-06-03]]). One finding reorders the whole decision and is recorded here so
the backend boundary is reviewable **before** code lands.

## Decision Drivers

- **Segment-level timestamps are non-negotiable.** The route emits `## [mm:ss]` segment anchors,
  and those timestamps are *also* the hook the Phase-2 companion-merge aligns on. A backend that
  cannot emit segment timestamps on the path we'd actually use is disqualified for v1.
- **Local-first / fully offline** (the air-gap test, VISION principle 1) — no cloud ASR API.
- **12 GB single GPU** — must fit beside (or time-share with) the existing resident set.
- **French-first multilingual** — French lecture audio is the real target, not English benchmarks.
- **Batch friction** — a user transcribes *many* lectures; per-file overhead matters.
- **Grounded HARD gate preserved** — `refusal_cf=1.0` / 0-hallucination is untouchable; audio is a
  parse-stage perception input, never a new grounding path.
- **Reproducible re-parse** — content-addressed `chunk_id`s require deterministic-or-cached output.
- **Minimize runtime sprawl** — adding inference runtimes has a real maintenance/VRAM/ABI cost.
- **Don't preclude the companion-merge** — v1's metadata must carry the alignment hooks.

## Considered Options

The decisive gate is **"can it emit segment timestamps on the path we'd run?"** — verified in
**vLLM source**, not vendor claims:

1. **Qwen3-ASR-1.7B served via vLLM `/v1/audio/transcriptions`** (the "reuse our serving seam"
   idea). **REJECTED for v1.** The endpoint returns **text only**; a `verbose_json` request is
   *rejected* (`"Currently do not support verbose_json for {model}"`). Timestamps gate on a
   per-model flag `supports_segment_timestamp` that **defaults False**; Whisper sets it True,
   Qwen3-ASR's vLLM class never does, and the verbose builder parses **Whisper-style `<|0.00|>`
   tokens Qwen does not emit** — which is exactly *why* a separate ForcedAligner exists.
2. **Qwen3-ASR + Qwen3-ForcedAligner-0.6B** (in-process or vLLM pooling endpoint). Gives word/char
   timestamps but is a **2-model, 2-pass** pipeline, **off the HTTP seam**, with the aligner
   **hard-capped at 300 s / 5 min per call** → a 1–2 hr lecture becomes ≥12–24 stitched chunks.
   Deferred (revisit only if Qwen's transcription accuracy proves decisive on French).
3. **Whisper-large-v3 served via vLLM `/v1/audio/transcriptions`.** Returns `verbose_json` with
   **segment timestamps** (vLLM PR #24209, present in our pinned `vllm>=0.21,<0.22`); French via
   `language='fr'`. **But:** **no word timestamps** (`words` always null, deferred to WhisperX,
   vLLM #25750), **no built-in VAD** (hallucination risk on silent passages), and a reported
   on-Ada WER regression in some versions (#33107 — on an L40S, the same Ada/sm_89 tier as
   the 4070, on 0.14.1) → a **WER spot-check on our pinned
   serve is mandatory**. Kept as a **first-class config-selectable backend** (zero new runtime,
   in-stack) but carries the per-doc cold-start tax (it is a *Whisper* model, so it cannot share
   the *Qwen* orchestrator process — it needs a short-lived parse-time serve like the VLM).
4. **Whisper via HF transformers** (the `automatic-speech-recognition` pipeline, in-process).
   **Zero new runtime** — reuses the exact transformers+torch stack Memex already runs for
   Nemotron chart-OCR and OTTER NER. Segment + word timestamps; long-form via `chunk_length_s`
   (HF flags it "very experimental"). Loads **once** per batch.
5. **faster-whisper (CTranslate2)** — the *same* Whisper-large-v3 weights via the CTranslate2
   runtime: ~same accuracy, ~4× faster, ~half the VRAM (int8 ~2.9 GB), with **native Silero VAD +
   native long-form + word timestamps + reproducible greedy decode**. Adds CTranslate2 as **one
   new (bounded) perception-model runtime**. **The recommended v1 default**, pending the A/B.
   WhisperX layers wav2vec2 word-alignment + pyannote diarization on top (up to 3 runtimes) —
   adopt its *method* (VAD-chunk, forced-align) later, not its full stack up front.
6. **NVIDIA Parakeet-v3 / Canary-1b-v2** — best *clean* French WER, but 12 GB long-form friction
   (Parakeet needs a local-attention switch) and NeMo-framework heaviness (Canary). A pilot
   candidate, not the default. (⚠️ Parakeet-**v2 is English-only** — a trap; v3 added EU langs.)
7. **whisper.cpp (ggml)** — lightest offline runtime; another distinct runtime; weaker word
   timestamps. A low-resource fallback, not the default.

**Disqualified:** Voxtral (the ~30-min long-form cap — Voxtral Mini is open-weights/Apache-2.0, so
it's the cap, not availability, that rules it out), Phi-4-MM (~40 s ASR cap), the canonical
distil-whisper English checkpoints (English-only — the *distillation technique* is not; the
recommended FR default is itself a Whisper distillation), Moonshine (no official French).

## Decision (Proposed)

**Add an audio ingestion route as a new parse path, with a PLUGGABLE ASR backend.** The route —
detect audio → VAD-chunk → transcribe → assemble **timestamped Markdown** → existing pipeline — is
**fixed and backend-agnostic** (the durable 90%). The **backend is config**
(`ModelSettings.asr` + `asr_backend: Literal["faster_whisper", "vllm", "transformers"]`, mirroring
`vlm_serving` / `enrich_ner_backend`), so the engine choice is swappable without touching the route.

- **Recommended v1 default backend: `faster_whisper`** running a French-capable Whisper-large-v3
  build (stock `large-v3` / `large-v3-turbo`, or the French-distil `bofenghuang/whisper-large-v3-french-distil-dec16`,
  MIT, which explicitly reduces long-form hallucination). **Final default is gated on a hands-on
  French-audio A/B on the user's own lectures** (§Revisit) — this ADR names the recommendation, not
  an irreversible default.
- **Whisper-via-vLLM** and **Whisper-via-transformers** ship as first-class config alternatives.
- The transcript is its **own content-only vault document** (ADR-0003), grounded on its text. The
  **companion-document merge is deferred to Phase 2** but the v1 metadata contract carries its hooks.
- **Transcript chunks are CLEANED, not dumped raw.** Spontaneous speech is non-linear (fillers,
  restarts), so a **deterministic** normalization (`core/text.normalize_transcript_text`: strips only
  non-lexical fillers — EN+FR `um`/`uh`/`euh` — + whitespace/punctuation artifacts, **preserving all
  LEXICAL content** [whole filler tokens only; a rare capitalised filler-homograph is the bounded
  exception], with the verbatim raw kept in the ASR cache as the faithfulness anchor) runs at assembly,
  keeping retrieval clean + reproducible. A heavier LLM **"structuring"** pass (paragraphing / run-on
  splitting) is the deferred immediate follow-up, gated behind a faithfulness guard + eval (spec
  §"Transcript normalization").

**This does NOT conflict with ADR-0001.** ADR-0001 commits the **agentic generation/orchestration
engine** (structured-output reliability under 10+ sequential calls per query) to vLLM. ASR is a
**parse-stage perception model** — the same category as the embedder (sentence-transformers),
the reranker (sentence-transformers), Nemotron chart-OCR (in-process transformers), and OTTER NER
(in-process transformers), **none of which run on vLLM**. In-process is in fact the *light default*
for perception models here; VLM-via-vLLM (ADR-0006 §4) was a *reluctant, quant-forced* exception
that carries a **~30 s cold-start per document** (`vlm-vllm-serving.md`), and that tax **does not
transfer** to an
in-process ASR backend (it loads once and stays resident across a batch). So ADR-0001 is **neutral**
on the ASR backend; it neither bars faster-whisper nor favors a vLLM-served ASR. The one bounded
cost faster-whisper adds — **CTranslate2 as a new runtime** — is justified by the timestamp gate
above (Whisper-on-vLLM lacks VAD/word-ts; the transformers long-form path is "experimental") and is
escape-hatched by the pluggable backend (`asr_backend="transformers"` adds **zero** new runtime).

## The timestamp gate (the load-bearing rationale, source-proven)

| Path | Segment timestamps over the path we'd run? | Word ts? | Runtime | Per-doc cold start |
|---|---|---|---|---|
| Qwen3-ASR via vLLM `/v1/audio/transcriptions` | **NO** (rejected) | NO | vLLM | yes (short-lived serve) |
| Whisper-large-v3 via vLLM `/v1/audio/transcriptions` | **YES** (segment only, PR #24209, in our 0.21) | NO | vLLM | yes (short-lived serve) |
| Whisper via HF transformers (in-process) | YES | YES (approx) | transformers (**0 new**) | no (load once) |
| **faster-whisper (CTranslate2)** | **YES native** | YES | CTranslate2 (**+1**) | no (load once) |
| Qwen3-ASR + ForcedAligner | YES (off-HTTP, 2-pass) | YES | vLLM-pooling / transformers | yes |

The pre-research document's "serve the ASR model on vLLM like everything else" framing is **false
for Qwen3-ASR** (text-only over HTTP) and **only partly true for Whisper-on-vLLM** (segment yes,
word no, VAD no). The elegant-seam intuition does **not** survive contact with the source.

## Consequences

### Positive

- A lecture MP3 becomes a first-class, searchable, **grounded-answerable** vault document with no
  change to the answering graph or its HARD gates.
- The route is backend-agnostic, so the A/B (and any future ASR SOTA shift) is a **config swap**,
  not a rewrite.
- Reuses the proven parse machinery: the suffix-dispatch precedent (`office`/`scan`), the
  `pause_vllm_for_gpu` GPU handoff, the content-addressed cache pattern, and the
  `page_char_counts → Chunk.page` attribution generalized to a **time range**.
- The Phase-2 companion-merge is *enabled, not foreclosed* — v1 carries the alignment metadata.

### Negative / Trade-offs

- The recommended default adds **CTranslate2** as a new runtime + a small dependency surface
  (`faster-whisper`, an audio decoder, Silero VAD). Mitigated by: it's a perception model (not the
  generation engine), it's ABI-independent of the torch/vLLM wheels, and `asr_backend="transformers"`
  is a zero-new-runtime fallback.
- **Offline provisioning caveat** — the ASR weights (and Silero VAD) must be downloaded **before
  going air-gapped** (same one-time online step as pyannote/OTTER). After that the route is fully
  offline.
- **No diarization in v1** — "who spoke" is a separate deferred follow-on (pyannote, HF-gated). Low
  value for single-instructor lectures (we ground on transcript *text*, not speaker labels).
- **Backend quality on spontaneous French + FR/EN code-switching is unmeasured by any public
  benchmark** — only the user's own A/B settles it (see §Revisit). Clean-speech WER rankings are
  known to **invert** on spontaneous speech (arXiv:2508.21193, where Whisper-large-v3 ranked 1st).

### Neutral

- `/ask`, `summarize`, chat, the bridge, MCP, and their HARD gates are byte-untouched (audio is a
  parse-stage addition; the answering side sees an ordinary transcript document).
- VISION's "Markdown as source of truth" holds — the canonical transcript `.md` is content-only;
  timestamps/segment metadata are **derived state in the manifest sidecar** (the `chart_extractions`
  precedent, ADR-0003).

## Companion-document merge (deferred to Phase 2, designed-for in v1)

The v1 route must **preserve the hooks** so the later merge is purely additive:

- **Keep the transcript as its OWN vault document, cross-linked to the slide PDF — do NOT merge
  inline.** Per ADR-0003 the canonical `.md` stays content-only; the alignment is **derived sidecar
  state**. Reuse `core/wikilinks.py` `[[doc#section]]` + the CITES edge + `related_documents`.
- **Minimal metadata contract v1 must satisfy:** per **segment** — (1) **global** start/end seconds
  vs the whole file (the thing the merge aligns on), (2) the segment's char-span in the transcript
  `.md`, (3) a language tag (FR/EN); and per **document** — (4) a stable link from the transcript doc
  to its companion deck `doc_id`, established at merge time via a `[[doc]]` / CITES edge (**NOT** a
  per-segment field — a standalone v1 ingest has no companion yet, and the merge aligns on the
  time-range + char-span, not a per-segment back-pointer).
- **The Phase-2 alignment method (adopt the method, not a new stack):** for each transcript chunk's
  time-range, find the slide/page whose text best matches by **EmbeddingGemma cosine** (per MaViLS,
  arXiv:2409.16765, OCR/text features contribute the most to slide↔transcript alignment — its
  multimodal pipeline reaches ~0.82 accuracy vs 0.56 for a SIFT baseline — while audio-transcript features
  alone still add value when OCR is sparse), optionally DP-regularized to be monotonic in time; the
  stored time-range narrows the candidate slides → cheap. Reuse Memex's own embedder + the deck's
  existing VLM-transcribed page text — do not import external models.

## Alternatives in Detail

### Serve the ASR model on vLLM to "reuse the seam"

The intuitive path, and the one this ADR was initially drawn toward. It dies on the timestamp gate
for Qwen (text-only over HTTP) and is only partial for Whisper-on-vLLM (no VAD / no word ts), which
*also* pays the per-doc cold-start because a Whisper model can't share the Qwen orchestrator process.
Kept as a config option, not the default.

### Make `faster_whisper` an irreversible hard default now

Rejected: spontaneous-French + FR/EN code-switch quality is genuinely unmeasured, and clean-WER
rankings invert on spontaneous speech. The honest position is a *recommended* default behind a
pluggable backend, with an A/B as the tie-breaker.

### Reuse Docling/whole-doc-VLM machinery for audio

Category error — audio has no pages to rasterise. The scan→VLM route is the structural *analogue*
(whole-source transcription bypassing the page pipeline), not a literal reuse.

## Amendment (2026-06-03): audio-bearing VIDEO ingestion ("class video", standalone)

The user's actual class recordings are **`.mp4` ZOOM videos** (each ~2.5–3 hr, `ftypisom` major brand,
an AAC stereo audio track + a video track), not audio files — exactly the "class video" case §5
originally deferred. The original "audio-only ingest; video is a Phase-2 audio-extraction extension"
scoping is **promoted now**, because the extraction is trivial: faster-whisper's PyAV decoder reads the
container's audio stream directly (no separate ffmpeg step), so a video container needs no new
transcription machinery — only the **gate** had to open.

- **`VIDEO_SUFFIXES = {.mp4, .m4v, .mov, .webm, .mkv}`** route to the **same `_parse_audio`**
  (`MEDIA_SUFFIXES = AUDIO_SUFFIXES | VIDEO_SUFFIXES`). The **visual track is ignored in v1** — the
  transcript is audio-only; the slide content comes from the companion PDF via the (still-deferred)
  Phase-2 merge above, for which these recordings + their `Cours N.pdf` decks are the concrete input.
- **Ingest acceptance** gains a `"video"` `DetectedKind` + `_detect_video`: a **curated** ISO-BMFF
  `ftyp` VIDEO-brand set (so HEIC/AVIF **image** brands stay rejected) or the Matroska/WebM **EBML**
  magic. A video with **no** audio track transcribes to nothing → recoverable refuse — **HARD-gate-safe
  by the same construction as an empty audio transcript.** Parse-stage only ⇒ the gate posture is
  unchanged.
- This is **standalone** video transcription only. The companion-MERGE (aligning the transcript's
  time-ranges to the slide deck) remains the Phase-2 deferral above; the per-segment timestamp + the
  document-level link hooks already carry it.

## Revisit When

- **The French-audio A/B runs** (the gating experiment): on a representative sample of the user's
  own lectures, race `faster_whisper` [stock `large-v3-turbo` + `bofenghuang` French-distil],
  `Whisper-via-vLLM` [pinned 0.21 serve, segment-only, WER spot-checked], and `Parakeet-v3`
  [transformers path] on spontaneous-FR WER + FR/EN code-switch + long-form hallucination + timestamp
  usability + runtime-count friction. **Only then** is the default crowned → move Status to Accepted
  and record the realized backend.
- **An independent French/Open-ASR-Leaderboard row** lands for Qwen3-ASR (today its SOTA WER is
  vendor-asserted; replication pending) — re-weigh the Qwen+ForcedAligner path.
- **The companion-merge is built** — record the alignment design as its own spec amendment and add
  the diarization follow-on if multi-speaker recordings enter scope.
- **vLLM's Whisper path gains word timestamps / built-in VAD** (#25750) — re-weigh
  `asr_backend="vllm"` as the in-stack default.

## References

- **Spec:** [`audio-asr-route.md`](../specs/audio-asr-route.md) — the implementation design.
- [ADR-0001](0001-vllm-as-sole-inference-engine.md) — vLLM as the **generation** engine (this ASR
  route is a perception model, in the embedder/reranker/chart-OCR/NER category, off that decision).
- [ADR-0003](0003-markdown-vault-as-source-of-truth.md) — content-only `.md` + regenerable derived
  sidecar state (the transcript + its segment-timestamp sidecar follow this).
- [ADR-0006](0006-cuda-dispatch-and-dtype.md) §4 (the P2.3 VLM-via-vLLM amendment) — the reluctant quant-forced exception +
  the parse-time serve lifecycle the `asr_backend="vllm"` option reuses (the ~30 s cold-start figure
  itself is in [`vlm-vllm-serving.md`](../specs/vlm-vllm-serving.md)).
- [ADR-0012](0012-otter-bert-ner-enrich-backend.md) — OTTER BERT-NER: the precedent for an in-process,
  off-vLLM perception model (lazy process-global, out of `models/registry`), the pattern the ASR
  backend follows.
- Specs [`scan-vlm-parse.md`](../specs/scan-vlm-parse.md) (the route analogue),
  [`vlm-vllm-serving.md`](../specs/vlm-vllm-serving.md) (the serve seam),
  [`vlm-transcription-cache.md`](../specs/vlm-transcription-cache.md) (the cache pattern).
- [[asr-audio-scope-2026-06-03]] — the 10-agent verification (the timestamp gate, the French
  inversion, the backend matrix); [[qwen-migration-research-2026-05-26]] — "the 4B does no audio →
  Whisper ASR Tier-1" origin. ROADMAP line ~140 (the scoped feature).
- vLLM PR #24209 (Whisper segment timestamps), issue #25750 (no word timestamps), issue #33107 (L40S/Ada
  WER regression on 0.14.1); arXiv:2508.21193 (spontaneous-French ranking inversion);
  arXiv:2409.16765 (MaViLS slide↔transcript alignment).
