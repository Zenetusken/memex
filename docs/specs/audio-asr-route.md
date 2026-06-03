# Spec: audio → timestamped-Markdown ASR parse route

**Status:** **Proposed** (design — NOT built). **ADR:** [ADR-0017](../adr/0017-audio-asr-ingestion-route.md).
**Analogue:** `scan-vlm-parse.md` (whole-source transcription that bypasses the page pipeline).
**Research basis:** [[asr-audio-scope-2026-06-03]] (the timestamp gate + the French A/B).

> This spec designs the route so the **ASR backend is pluggable** and the **companion-document
> merge** (ADR-0017 §Phase-2) is enabled but not built. The recommended v1 default backend is
> `faster_whisper`; the **final default is gated on a hands-on French-audio A/B** (see §13).

## 1. Problem

Memex ingests PDF / Office / images / scans but has **no audio path**. The target is
**class-lecture audio** — MP3, 1–2 hr, multilingual/**French**, sometimes FR/EN code-switching. The
orchestrator (`Qwen3.5-4B`) and doc-VLM (`Qwen3-VL-8B`) are vision-language only (no `audio_config`)
→ audio needs a **dedicated ASR model**. The route must emit **segment-level timestamps**
(`## [mm:ss]` anchors) — for navigation *and* as the hook the Phase-2 slide-merge aligns on — then
hand a normal transcript document to the existing chunk/embed/retrieve/answer pipeline, HARD gate
intact.

## 2. The timestamp gate (why not "just serve it on vLLM")

Verified in **vLLM source**, not vendor claims (full record: [[asr-audio-scope-2026-06-03]]):

- **Qwen3-ASR via vLLM `/v1/audio/transcriptions` returns TEXT ONLY** — `verbose_json` is *rejected*
  (`"Currently do not support verbose_json for {model}"`). The timestamp path gates on a per-model
  flag `supports_segment_timestamp` (defaults False); Whisper sets it, Qwen3-ASR's vLLM class does
  not, and the verbose builder parses Whisper-style `<|0.00|>` tokens Qwen never emits (hence the
  *separate* ForcedAligner). So the elegant "reuse the `_serve_vlm_vllm` seam for Qwen" idea is dead.
- **Whisper-large-v3 via vLLM** *does* return `verbose_json` **segment** timestamps (PR #24209,
  present in our `vllm>=0.21,<0.22`), but **no word timestamps** (`words` null, #25750) and **no
  built-in VAD**; and a Whisper model can't share the *Qwen* orchestrator process → it needs its own
  short-lived parse-time serve (the per-doc cold-start tax). WER must be spot-checked on **our pinned
  0.21 serve** (an on-Ada WER regression was reported on 0.14.1 — #33107, on an L40S, the same
  Ada/sm_89 tier as the 4070).

⇒ The default backend is **in-process** (`faster_whisper`), with `vllm`/`transformers` selectable.

## 3. Decision — route design (backend-agnostic; the durable 90%)

A new `_parse_audio` route, modelled on `_parse_scan_with_vlm`'s write/manifest/return tail, with
the body produced by an ASR backend instead of the VLM:

```
parse_document(suffix in AUDIO_SUFFIXES)
  → _parse_audio(vault, doc_id, source):
        async with pause_vllm_for_gpu():           # orchestrator down → GPU free (nestable)
            segments = await transcribe_audio(source=source, cache=asr_cache, refresh=refresh_asr)
        body = assemble_transcript_markdown(segments)   # ## [mm:ss] headers + text
        _finalize_body(body) → write_document → update_manifest(parse=ParseStage(
              pages=[], segments=[TranscriptSegment...], ...))   # engine tag is on ParseResult, NOT ParseStage
        return ParseResult(doc_id, correlation_id, engine="asr", pages=[], markdown_bytes=…)
```

The **backend-agnostic transcription pipeline** inside `transcribe_audio` (the part that is identical
across every backend):

1. **Decode** the source to 16 kHz mono PCM (the audio decoder bundled with the backend, e.g.
   faster-whisper's `av`; no system ffmpeg dependency required).
2. **VAD once** over the whole file (Silero VAD) → speech regions. (faster-whisper does this
   internally; the `vllm`/`transformers` backends call a shared VAD helper so all three behave alike.)
3. **Merge/split** into safe-length chunks, cutting **only on silence** (never mid-word): ~30 s for
   Whisper-family backends.
4. **Transcribe** each chunk independently (batched within VRAM).
5. **Offset** every local timestamp by the chunk's absolute start: `global = chunk_start + local`.
6. **Normalize** each segment's text — `core/text.normalize_transcript_text`, a deterministic,
   faithful clean of non-lexical speech noise (gated by `ParseSettings.asr_normalize`, default on;
   see §"Transcript normalization" below).
7. **Assemble** `## [mm:ss]\n<text>` blocks in time order → the transcript body.

Chunking is **mandatory, not an optimization** — no backend transcribes a 1–2 hr file in one pass
(Whisper's window is 30 s; Qwen caps a single utterance at 20 min, its aligner at 300 s). Per-chunk
independent anchoring also prevents Whisper's **cumulative timestamp drift** (the buffered window
shifts by the *previously decoded* timestamp, so errors accumulate over hours). Peak VRAM is
**file-length-independent** (≈ chunk × batch), so a 2-hour file fits the 4070 at batch 4–8.

### Transcript normalization (faithful + deterministic)

Raw ASR output is spontaneous, non-linear speech — disfluencies, fillers, restarts — so the chunks
must be CLEANED before they reach the RAG pipeline, but **without losing context and 100%
deterministically faithful** (the grounding gate reads this text). Two tiers resolve that tension:

- **v1 — a DETERMINISTIC, content-preserving pass** (`core/text.normalize_transcript_text`, shipped
  with this route): a pure function that removes ONLY non-lexical noise — unambiguous filler
  interjections (`um`/`uh`/`euh`/…, EN+FR; backchannels like `uh-huh` and ambiguous markers like
  `like`/`so`/`ben` are EXCLUDED) plus whitespace/punctuation artifacts. **By construction it never
  alters a content word and never adds text**, so it is meaning-preserving ("100% faithful") and
  reproducible (byte-identical output → stable content-addressed `chunk_id`s). It runs at step 6,
  **after** the ASR cache, so the **verbatim raw transcript stays cached** (the audit/faithfulness
  anchor), toggling `ParseSettings.asr_normalize` re-cleans from cached raw (no re-transcribe), and it
  is **NOT** part of the cache `cfg` key. It deliberately does NOT collapse content-word stutters
  ("the the cat"), split run-on sentences, or restructure paragraphs.
- **Follow-up — an OPTIONAL LLM "structuring" pass** (deferred, §15): better paragraphing / sentence
  segmentation / light disfluency smoothing, which needs semantics. Because it would rewrite the
  grounding source-of-truth, it is gated behind (a) a **deterministic faithfulness guard** — the
  cleaned output's content tokens must be a subset of the raw's (no new content), falling back to the
  raw on violation (the Table-RAG-verbatim philosophy) — (b) a content-addressed cache for
  determinism, and (c) a transcript-fidelity eval. Its own arc.

## 4. Backend abstraction (`ModelSettings.asr` + `asr_backend`)

New on `ModelSettings` (`core/config.py`), mirroring `vlm` / `vlm_serving` and
`enrich_ner_backend`:

```python
# ASR (audio transcription) — the parse-time speech-to-text model. Off the
# grounded path (a perception model, like the embedder / reranker / chart-OCR /
# OTTER NER — none on vLLM); ADR-0017. Default backend is in-process
# faster-whisper (the timestamp gate rules out Qwen-via-vLLM; ADR-0017 §2).
asr: str | None = None   # the FR-capable Whisper-large-v3 build (A/B-gated, §13); typed like the
                         # deferred summarizer/reasoner swap-in slots — None → ASRUnavailable (§5)
asr_backend: Literal["faster_whisper", "vllm", "transformers"] = "faster_whisper"
asr_serve: ASRServeSettings = Field(default_factory=ASRServeSettings)  # only for asr_backend="vllm"
```

- `faster_whisper` — CTranslate2, in-process, loads once. Native VAD + long-form + word timestamps +
  reproducible greedy. **Recommended v1 default.** Recommended model: stock `large-v3` /
  `large-v3-turbo`, or `bofenghuang/whisper-large-v3-french-distil-dec16` (MIT, reduces long-form
  hallucination; ships CTranslate2 weights).
- `transformers` — the HF `automatic-speech-recognition` pipeline, in-process. **Zero new runtime**
  (reuses the Nemotron/OTTER transformers stack). Long-form via `chunk_length_s` (HF-flagged
  experimental). The zero-dependency fallback.
- `vllm` — a short-lived parse-time vLLM serving a **Whisper** build (a new `_serve_asr_vllm`
  cloned from `vlm_backend._serve_vlm_vllm`), transcribing over `/v1/audio/transcriptions`
  (`response_format="verbose_json"`). **Segment timestamps only**; inherits the ~30 s cold-start.
  `ASRServeSettings` mirrors `VLMServeSettings` (host/port distinct from the orchestrator + VLM).

The route reads `asr_backend` and dispatches; everything else (VAD, chunking, offsetting,
assembly, cache, manifest) is shared.

## 5. Detection & routing

**Ingest acceptance** (`ingest/validation.py`): add an `"audio"` member to `DetectedKind`. **Note
the existing `_MAGIC` mechanism only matches at byte offset 0** (`head.startswith(prefix)`): `ID3`,
the `\xFF\xFB` MP3 frame-sync, `fLaC`, and `OggS` are offset-0 and fit as plain new `_MAGIC` rows,
but `RIFF….WAVE` (the `WAVE` tag is at byte 8) and `ftyp` (M4A/MP4 box-type, at byte 4) are **NOT**
offset-0 — they need either an offset-aware extension to `_MAGIC`/`_detect` or a dedicated branch in
`_detect` (the precedent is the hardcoded `docx → _refine_office` ZIP branch), not plain prefix rows.
The `audio` kind maps in `ingest/pipeline.py::_EXTENSION_FOR_KIND` (e.g. `{"audio": ".mp3"}` keyed off
the detected container, or preserve the original suffix). Magic-number validation stays
non-optional (GUIDELINES Part VI) — a new format arrives **with** this ADR per the validation
docstring.

**Parse dispatch** (`parse/pipeline.py::parse_document`, the suffix switch ~2288–2311): add a branch
**before** the PDF branch, exactly mirroring the `OFFICE_SUFFIXES` branch:

```python
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
...
if source.suffix.lower() in AUDIO_SUFFIXES:
    return await _parse_audio(settings.vault_path, doc_id, source, refresh_asr=refresh_asr)
```

Gating: there is **no `disable_asr` fallthrough** (unlike `disable_vlm`) — an audio file has no
non-ASR parse, so if the backend is unavailable **or no `asr` model is configured** the route raises
a recoverable `ASRUnavailable` (a new `MemexError` subclass in `parse/asr_backend.py`, per the
error-typing rule), surfaced to the user. (Contrast the scan route, whose `disable_vlm=True` default falls through to Docling-OCR.)

## 6. `_parse_audio` + the backend module

New `parse/asr_backend.py` (the audio sibling of `vlm_backend.py`):

- `async def transcribe_audio(*, source: Path, cache: ASRTranscriptionCache | None = None,
  refresh: bool = False) -> list[TranscriptSegment]` — the backend-agnostic pipeline of §3, returns
  time-ordered segments. Per-chunk failures are recorded as a segment with `confidence=0.0` + the
  error in `rationale` (never a silent drop — the scan-route contract).
- `ASRUnavailable(MemexError)`, `ASRTranscriptionError(MemexError)` — typed, `context`-carrying.
- The `vllm` backend's `_serve_asr_vllm` reuses the `_serve_vlm_vllm` lifecycle verbatim (spawn-time
  gid capture, group-emptiness reap, startup retry, stderr capture — `vlm-vllm-serving.md` §Lifecycle).

`_parse_audio` (in `pipeline.py`, modelled on `_parse_scan_with_vlm`): open the ASR cache,
transcribe under `pause_vllm_for_gpu()`, assemble the body, `_finalize_body` → `write_document` →
`update_manifest(parse=ParseStage(pages=[], segments=…, figure_count=0, …))` — like the scan route,
the `ParseStage` carries **no** `engine` field; the engine tag lives on `ParseResult` only — return
`ParseResult(engine="asr", pages=[], markdown_bytes=…)`. No chart-OCR pass (no figures).

## 7. Transcript Markdown shape

```markdown
# <lecture title>

## [00:00]
<verbatim transcript text for this segment>

## [00:42]
<…>
```

`## [mm:ss]` (or `[hh:mm:ss]` past an hour) headers are the **time anchors** — real Markdown
headings, so they become `heading_path` entries and section boundaries in the existing chunker (a
transcript chunk inherits its segment's timestamp via its heading). The body is content-only (ADR-0003);
the machine-readable timestamps live in the sidecar (§8). The title comes from the source filename
(a future LLM-titling pass could improve it; `index/pipeline.py::retitle_document` is the existing
deterministic *writer* that propagates a new title across the denormalized stores once a title string
exists — it does not itself generate one).

## 8. Metadata contract — `TranscriptSegment` on `ParseStage`

A new sidecar list on `ParseStage` (`core/manifest.py`), the **`chart_extractions` precedent** (a new
optional typed list; legacy manifests load unchanged; empty default is a no-op):

```python
class TranscriptSegment(BaseModel):
    index: int            # 0-based, document order
    char_start: int       # span in the canonical .md (join-time, pre-transform)
    char_end: int
    start_s: float        # GLOBAL seconds vs the whole file — the Phase-2 alignment key
    end_s: float
    language: str = ""    # detected/forced lang tag (FR/EN…)
    confidence: float = 1.0
    rationale: str = ""   # failure note for a dropped chunk (confidence=0.0)

class ParseStage(BaseModel):
    ...
    segments: list[TranscriptSegment] = Field(default_factory=list[TranscriptSegment])
```

**Audio does NOT reuse `PageDecision`/`pages`** (audio has no pages; the `engine` Literal is *not*
extended with `"asr"`). `ParseStage.pages` stays `[]` for audio; `ParseResult.engine` is the free
`str` `"asr"`. The dedicated `segments` list keeps the time/char/lang record audio-native and
PageDecision unpolluted.

**Chunk time attribution** generalizes the proven `page_char_counts → Chunk.page` machinery
(`index/chunker.py::page_intervals` / `_page_for_offset`, ADR/CLAUDE "navigation-grade, not
citation-grade"): `index_document` passes the segments' `(char_start, char_end, start_s, end_s)` to
the chunker, which locates each chunk's `char_start` in the intervals and stamps a new optional
field `Chunk.time_range: tuple[float, float] | None = None` (`core/types.py`, additive,
backward-compatible — the doc/PDF paths leave it `None`). The webui surfaces it as a `[mm:ss]` label
on each source chip (the audio analogue of the `?page=N` jump), and Phase-2 alignment reads it.

Note the companion-deck link is **document-level** (a `[[doc]]`/CITES edge between the transcript doc
and the slide doc), **not** a `TranscriptSegment` field — a standalone v1 ingest has no companion, and
the Phase-2 merge aligns on the time-range + char-span, not a per-segment back-pointer. The
per-segment record carries only timestamps + char-span + language.

## 9. ASR transcription cache (reproducibility)

Mirror `parse/vlm_cache.py::VLMTranscriptionCache` exactly — a new
`parse/asr_cache.py::ASRTranscriptionCache` (sqlite at `vault/.memex/asr_cache.sqlite`,
`isolation_level=None`, `apply_sqlite_pragmas`, `asyncio.Lock`-gated writes, `INSERT OR IGNORE`,
`get`/`put`/`delete_by_audio`/`close`, in the `reindex --force` teardown).

**Two key differences from the VLM cache.** The VLM cache keys on `sha256(pdf_bytes):page:model:prompt`,
where `page` is INPUT-derived (known before the call). ASR differs twice:

1. **The cache unit is the VAD CHUNK, not a "segment".** Segments are OUTPUT-derived (a VAD chunk
   emits ≥1 sub-segments only *after* transcription), so a segment index can't key a lookup that runs
   *before* the transcription it is meant to skip. But VAD is a **deterministic function of
   `(audio_bytes, vad_params)`**, so the chunk boundaries — hence `chunk_index` — ARE known
   pre-transcription (§3 steps 2–3). The cached value is that chunk's emitted segments (text +
   **chunk-local** timestamps); a re-parse re-runs the deterministic VAD → same chunks → per-chunk
   cache hits → reassemble (re-applying each chunk's absolute offset, itself fixed by the VAD).
2. **The key carries a decoding-param `cfg`** (ASR has no prompt, but decoding params that change the
   output):

```
sha256(audio_bytes) : chunk_index : model : cfg
   where cfg = sha8(json{backend, beam_size, language|"auto", temperature, vad_params, chunk_window})
```

**Without `cfg` a decoding-param change would silently REPLAY stale output instead of a clean miss.**
Greedy ASR (beam=1, temp=0) is reproducible for fixed input+hardware+library-version but **not
bit-exact across CUDA/cuDNN/lib upgrades**, so — exactly like the VLM — the cache (not decode
determinism) is the reproducibility guarantee that keeps content-addressed `chunk_id`s stable across
re-parse. A `_MIN_CACHEABLE_CHARS`-style guard avoids freezing an empty/failed chunk — applied in
`transcribe_audio` **before** `cache.put` (the guard lives in the CALLER, mirroring `convert_pages`
in `vlm_backend.py`, **NOT** in the cache class). `memex parse --refresh-asr <doc>` busts one document.

## 10. GPU lifecycle

ASR runs at **parse time under `pause_vllm_for_gpu()`** (orchestrator down → card free), nestable so
the CLI `ingest`/`index`/`reindex` outer pause makes the inner one a no-op (one pause/restart for the
whole run). The in-process backends (`faster_whisper`/`transformers`) load **once** and stay resident
across a batch — **no per-doc cold-start** (the key win over the `vllm` backend, which clones the
short-lived-serve lifecycle and pays ~30 s/doc). Co-residence with the orchestrator is **not** the
constraint (the orchestrator is paused); startup + runtime are — which is why in-process is the
default. VRAM fit: faster-whisper int8 ~2.9 GB / Whisper-v3 fp16 ~3.6 GB, trivially within the freed
card.

## 11. HARD-gate posture

Parse-stage only ⇒ **HARD-gate-neutral by construction**. The route grounds answers on the
**transcript text** like any document. All-fail (every chunk errors) → empty body → 0 chunks →
answer/summarize **REFUSES** (the scan-route precedent; never fabricates from an unreadable file).
`/ask`, `summarize`, chat, bridge, MCP, and their gates are byte-untouched.

## 12. Config, dependencies, surfaces

- **Dependencies** — a new `[project.optional-dependencies] audio = [...]` extra (`faster-whisper`,
  its audio decoder, Silero VAD). CTranslate2 ships its own CUDA libs (ABI-independent of the
  torch/vLLM cu129 wheels — a pro). `asr_backend="transformers"` needs only the existing `models`
  extra (zero new deps). Document the install in `docs/deploy/`.
- **Offline provisioning** — download the ASR model + Silero VAD **before air-gapping** (the
  one-time online step, same as pyannote/OTTER); afterward the route passes the air-gap test.
- **Surfaces** — a transcribed lecture is an **ordinary document**, so `ask` / `summarize` / `chat` /
  the webui doc-view all work with no new surface. CLI `memex ingest lecture.mp3` just works once the
  route lands. (A future webui audio player synced to the `## [mm:ss]` anchors is a nice-to-have, not
  v1.)

## 13. Eval — transcription fidelity + end-to-end

- **Parse-fidelity (WER/CER):** reuse `memex eval-parse` (WER via the in-house
  `eval/scoring.py::word_error_rate`; `jiwer` is present in the `eval` extra if a library swap is
  wanted, but is not currently wired in) — a small `tests/eval-data/audio-*/` corpus of the user's own clips with hand-checked
  `ground-truth.md`, scoring WER against the transcript body (strip the `## [mm:ss]` anchors before
  scoring). **Clips stay LOCAL** (the eval-corpus convention).
- **End-to-end answer-eval:** a query set over a transcribed lecture, run through `memex eval` — must
  hold `refusal_cf=1.0` / 0-hallucination like every other corpus.
- **The backend A/B (the gating experiment, ADR-0017 §Revisit):** on a representative sample of the
  user's own French lectures, race `faster_whisper` [stock `large-v3-turbo` + `bofenghuang`
  French-distil], `Whisper-via-vLLM` [pinned 0.21 serve, WER spot-checked], and `Parakeet-v3`
  [transformers path], scoring **spontaneous-FR WER + FR/EN code-switch handling + long-form
  hallucination + timestamp usability + runtime-count friction**. Clean-WER rankings invert on
  spontaneous speech (arXiv:2508.21193) → **read-speech benchmarks are not decision-grade**; the A/B
  on real lecture audio settles the default.

## 14. Testing

- **Unit** (`test_asr_route.py`): the pure VAD-chunk → transcribe → **global-offset** → assemble
  assembly from a **faked** backend (no GPU); a failed chunk → recorded segment, `confidence=0.0`, no
  silent drop; the `## [mm:ss]` formatting (incl. `hh:mm:ss` past an hour); the cache key includes the
  decoding-param `cfg` (a param change ⇒ a miss).
- **Unit** (`test_chunker.py` additions): `Chunk.time_range` populated from segment intervals via the
  generalized `_page_for_offset` (and `None` when no segments — back-compat).
- **Integration** (`test_audio_routing.py`, the `test_scan_routing.py` sibling): a faked
  `transcribe_audio` + an audio-magic source → `parse_document` routes to `_parse_audio`, writes the
  timestamped body, records `segments` (not `pages`) in the manifest with `engine="asr"`; the
  `AUDIO_SUFFIXES` gate; `ingest/validation` accepts the audio magic + rejects a bogus one.
- **Live (GPU)**: ingest a real French lecture clip → transcribe → chunks index → `memex ask` answers
  from the transcript → the answer's source chip shows a `[mm:ss]` anchor. (Then the audio eval corpus.)

## 15. Out of scope (deferred)

- **Diarization** ("who spoke") — pyannote, HF-gated; low value for single-instructor lectures (we
  ground on text). A follow-on if multi-speaker recordings enter scope (ADR-0017 §Revisit).
- **The companion-document merge** — ADR-0017 §Phase-2: align the transcript's per-segment time-ranges
  to a slide deck/PDF's pages via EmbeddingGemma cosine (MaViLS method), as a cross-linked sidecar.
  v1 only **preserves the hooks** (per-segment global timestamps + char-spans + language; the
  transcript↔deck link is a **document-level** `[[doc]]`/CITES edge established at merge time).
- **Qwen3-ASR + ForcedAligner** (word/char timestamps) — revisit only if an independent French result
  makes Qwen's accuracy decisive (the 2-pass, 300 s-capped, off-HTTP cost is otherwise not worth it).
- **An LLM "structuring" pass** over the deterministically-normalized transcript (the immediate
  follow-up to v1's deterministic clean) — paragraphing / run-on splitting / light disfluency
  smoothing that needs semantics — gated behind a **deterministic faithfulness guard** (cleaned
  content tokens ⊆ raw; fall back to raw on violation), a content-addressed cache, and a
  transcript-fidelity eval (§"Transcript normalization"). v1 ships only the deterministic pass.
- **A synced audio player UI** and an **LLM-titled transcript pass**.
