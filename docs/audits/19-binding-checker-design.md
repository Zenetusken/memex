# Content-class binding checker — buildable design (ar-12 fine-tune increment)

**Target failure class:** binding fabrication (FRANK EntE/CircE-in-article): a TRUE predicate+value
from the cited chunk re-attributed to a CO-PRESENT wrong subject (ar-12: consolidated gross margin
71.1% rebound to "the Graphics segment"). Seven zero-shot arms are kill-tested dead on this class
(audit-18, `docs/audits/data-17-scope-calibration/`); the deterministic provenance backstop closes
tg-13 but not the content class. This doc specifies the banked fine-tune.

**Synthesized from:** R1 (LettuceDetect/TinyLettuce recipe, source-verified), R2 (minting design,
literature-verified), R3 (local feasibility, repo-verified), R4 (landscape sweep, clean negative).
Repo anchors below re-verified against source 2026-06-12 (calibration artifacts, probe-script
symbols, `provenance_scope_enabled` config.py:710, `_provenance_scope_violation` answering.py:2351,
`pause_vllm_for_gpu` parse/pipeline.py:875, HF cache contents).

---

## 0. Conflict resolutions (explicit)

| Conflict | Reports | Resolution + why |
|---|---|---|
| FR-capable base model | R4 recommends `lettucedect-210m-eurobert-fr-v1` (two-model EN+FR split); R1+R3 recommend `jhu-clsp/mmBERT-base` single model | **Believe R1+R3.** R1 *verified loadability*: EuroBERT requires `trust_remote_code=True` (`'eurobert' not in CONFIG_MAPPING_NAMES` at transformers 4.57.6) — an env-policy liability; mmBERT-base is native ModernBERT arch, MIT, **already in the HF cache** (verified), and the env already runs an mmBERT backbone in production (`whoisjones/otter-bi-mmbert`). R4's pick was framing-level, not loadability-verified. |
| Mint scale | R1: ~1.5–3k pairs (TinyLettuce scale); R2: 10–16K (target 12K) | **Believe R2's analysis, adopt R1's ladder discipline.** Our discrimination is strictly harder than TinyLettuce's (presence held constant; only binding varies) and needs the anti-shortcut classes (hard positives, unit-transform positives) R1 doesn't budget for. Start at 4K, ladder to 12K, stop early if dev F1 saturates (data is cheap to extend). |
| Train with daemon up or paused | R3: "trainable without pausing (~5.8 GB free)"; R2: "train under `pause_vllm_for_gpu`" | **Believe R2 (pause).** Phases are naturally sequential anyway — minting needs the daemon UP, training doesn't; pausing gives deterministic headroom for seq-4096 batches and removes the OOM variable. R3's headroom figure is plausible but unmeasured under training allocator churn. |
| LLM perturber (`rag-fact-checker` RELATIONAL/FACTUAL) as the negative generator | R1+R4: usable, point `base_url` at local vLLM; R2: deterministic/structural minting with the 4B only as extractor/paraphraser | **Believe R2 as primary, R1/R4 as secondary.** R4 itself flags (measure-don't-assert) that the perturber's swap QUALITY on the ar-12 shape (co-present re-attribution, not subject-absent substitution) is unverified, and labels must be by construction. Deterministic swap + table structure guarantees the presence-preserving invariant; the LLM perturber is an optional diversity supplement whose outputs must pass the same F1–F5 filters and presence post-check. |
| Continued-FT from `lettucedect-base-modernbert-en-v1` vs fresh head | R1: lowest-variance for EN (cleanest FP profile, 9/12 flag nothing); R3: mmBERT-base natural base | **Primary = mmBERT-base fresh** (single FR+EN model — the gate itself contains a French FP, cr350-xref-02). **Fallback arm = EN continued-FT + RAGTruth replay** if mmBERT underperforms or its tokenizer-offset probe fails (§3 P0). |
| Checker invocation cost | R3: "one ≤4096-token forward per fired /ask" on CPU | Kept, but latency number is **HYPOTHESIS** until measured (§5). |

---

## 1. DECISION: base checkpoint + architecture

**Base:** `jhu-clsp/mmBERT-base` (307M, MIT, 8192 ctx, native `ModernBertForTokenClassification`
in transformers 4.57.6, already in the local HF cache — air-gap-ready, zero downloads).

**Architecture:** LettuceDetect-style token classification, `num_labels=2`
(`0=supported, 1=hallucinated`), char-span labels on the answer region, prompt loss-masked with
`-100`. Inference = the existing `lettuce_arm` recipe (`scripts/scope_guard_span_probe.py:546-658`):
`_LD_QA_TEMPLATE`-formatted prompt + answer as sentence pair, `truncation="only_first"`,
`max_length=4096`; **case score = max class-1 softmax over a contiguous class-1 run** — the same
thresholdable scalar (`ld_max_conf`) already recorded for the pip checkpoint in
`scope_probe_lettuce.json`, so candidate rows drop into the existing report machinery unchanged.
Training data is emitted in BOTH label formats (token-span + pair-level binary, per R2) so a
sequence-classification fallback head stays open at zero re-mint cost.

**The FR story:** the frozen gate includes a French FP (cr350-xref-02) and the vault has substantial
FR content (cr350, french-course). One multilingual model beats an EN+FR pair: mmBERT covers 1833
languages natively, loads without `trust_remote_code`, and matches the production OTTER precedent.
FR training signal = `KRLabsOrg/ragtruth-fr-translated` (HF, MIT, 17,790 rows, already in the exact
`HallucinationSample` schema — verified by R1) + ~10–15% FR in-domain mints (§2). The EuroBERT FR
checkpoint is rejected for `trust_remote_code`; the EN-only lettucedect checkpoint alone under-covers
the gate.

**Evidence trail:**
- Pip `lettucedetect==0.1.8` ships the full training stack as library code (Trainer,
  HallucinationDataset, evaluator) — verified in the local uv cache
  (`/home/drei/.cache/uv/archive-v0/DiSmg5FZr2zmpe92C7t8d/`); CLI scripts are repo-only but trivial.
  We **vendor ~300 lines (MIT), not pip-install** (the pip install previously broke the project torch
  combo — documented in `scope_guard_span_probe.py:557` docstring).
- TinyLettuce precedent: 3K in-domain synthetic trains 17–68M encoders to ~89–93 F1 in-domain —
  "specialized in-domain data beats parameter count" (R2's anchor; matches the banked audit-18
  verdict).
- R4 landscape: **qualified clean negative** — no off-the-shelf ≤1B checkpoint prices binding
  (RAGTruth/ANLI/summarization regimes are all binding-blind by inheritance); HalluGraph (Dec 2025)
  independently confirms the class is real and similarity-blind but ships no checkpoint. The
  fine-tune is the right spend.
- Audit-18 measured the pip lettucedect checkpoint as binding-blind but with the **cleanest FP
  profile of all arms (9/12 flag nothing)** — the recipe's FP discipline is worth inheriting; only
  the binding signal is missing from its training data.
- HYPOTHESIS (R1 open question #4): mmBERT's Gemma-2 tokenizer behaves correctly with the
  `answer_start_token` heuristic (`hallucination_dataset.py:114-124`) at pair-encoding. **Must be
  probed before training** (§3 P0).

---

## 2. Minting pipeline spec

### 2.1 Sources (all repo-verified)
- **Chunks:** vault docs via the production text path — `index/pipeline.py::build_chunking_body`
  (chart-sidecar re-attach + GFM handling) → production chunker. 7,369 chunks / 177 docs available.
  Emit chunk text AS-IS (raw GFM + `[table-rows]`) — both classes must see production formats (the
  DeBERTa-MNLI GFM false-low lesson).
- **Entities:** OTTER (`enrich/ner_otter.py`, multilingual, cached), kinds from
  `enrich/entities.py:18`.
- **Tables:** `index/table_store.py::extract_tables` → `StoredTable` headers+rows (deterministic
  structured access, no new parsing).
- **LLM (extractor/paraphraser ONLY, never labeler):** the live 4B daemon via guided JSON
  (`models/client.py`; endpoint recipe = `_judge_call`, `scope_guard_span_probe.py:760`). The 4B is
  a measured-bad judge for this class (18/22 FP, `mentioned=true` on the breach) — **labels are
  100% by construction.**
- **Real-distribution positives (optional supplement):** the `capture` subcommand harvests live
  (question, summary, claims, cited-window) tuples; sweeping the 177 ANS eval queries ≈ 3 h serial
  (daemon up, retrieval CPU-pinned). Use for dev-realism, not as the bulk source.
- **EN/FR replay:** RAGTruth (preprocessed via vendored `preprocess_ragtruth.py`) +
  `KRLabsOrg/ragtruth-fr-translated` — keeps the general-hallucination skill and the clean-FP
  profile; the mints add the binding class.

### 2.2 Presence-preserving swap algorithm (the core invariant)
Every negative satisfies: (a) swapped-in subject S2 occurs **verbatim in the chunk**; (b) the
predicate+value is true in the chunk **of some other subject S1**; (c) the claim is false **only**
because of the S1→S2 rebind. Within a matched pair the lexical overlap with the context is
near-identical, so no presence feature separates the classes — the model can only reduce loss by
learning binding. (FactCC's entity swap already draws from the same source doc, i.e. is
presence-preserving by construction — R2's verified reading; the 7 arms died because no *zero-shot*
checker was ever trained on such minimal pairs.)

```
INPUT:  eligible_docs = vault docs MINUS HOLDOUT_DOCS
        (HOLDOUT = every doc whose chunk_id appears in any of the 14 calibration tuples:
         scope_probe_fp.json cited[].chunk_id + the two breach cited_suffixes)
OUTPUT: rows = (chunk_text, claim_text, label∈{OK,BREACH}, breach_span|None, meta)

for doc in eligible_docs:
  body   = build_chunking_body(doc)            # production text parity
  tables = extract_tables(doc.id, body)
  for chunk in chunker(body):
    ents      = otter_entities(chunk)          # person/org/place/concept/method/tool
    subjects  = dedupe(ents ∪ row_first_cells(tables∩chunk) ∪ headers(tables∩chunk)
                       ∪ heading_terms(chunk))
    if count(subjects with same-kind sibling) < 2: continue   # need a rebind target

    # A. positives (true-of-chunk)
    A1 prose: 4B guided-JSON extracts ≤3 atomic claims
       {text, subject_span, predicate, value_span}, subject ∈ subjects,
       value verbatim-locatable in chunk (deterministic check; drop else)
    A2 table: verbalize(row_label, header_j, cell_ij) via template bank (≥8/lang)
    A3 unit-transform positives: deterministic value rewrites
       ($130,400M↔$130.4 billion; 71.1%↔"71.1 percent"; FR decimal comma) — label OK
       (these ARE the calibration FP modes; they must exist as positives)

    # B. binding negatives (presence-preserving rebind)
    for c in claims:
      pool = [s2 in subjects: kind(s2)==kind(c.subject), s2≠c.subject,
              not coreferent(s2, c.subject)]                  # F3 alias guard
      for s2 in sample(pool, k≤2):
        neg = splice(c.text, subject_span→s2)                 # EN proper-noun path
            | llm_mask_refill(c.text, s2)                     # FR + common-noun path
              (4B: "replace the subject with S2; change NOTHING else", guided JSON)
        assert byte-diff(neg, c.text) covers EXACTLY the subject span    # F5
        if accidental_truth(chunk, tables, s2, c.predicate, c.value): discard  # F1+F2
        if nli_entails(chunk, neg) > 0.5: discard                              # F4
        emit (chunk, neg, BREACH, span_of(s2))
        # span = positional from the splice/refill diff — NEVER .find()
        # (R1 open-q #2: the upstream .find() first-occurrence bug)

    # C. hard positives (kill the inverse-presence shortcut)
    for each s2 ever used as a rebind target:
      mint a TRUE claim ABOUT s2 (its own row / its own sentence), label OK

    # D. style decorrelation
    4B-paraphrase a random 50% of BOTH classes (same prompt/temp);
    re-run F1–F5 on paraphrased negatives; re-verify value verbatim for positives

balance 1:1; caps ≤6/chunk, ≤400/doc, ≤15%/template; shuffle; split BY DOCUMENT
```

### 2.3 Table-row/column swap special case (the literal ar-12 generator)
TabFact precedent: same-column swap is presence- and type-preserving by construction.
- **NEG-row:** cell (i,j) re-attributed to row k's label. Guard `cell(k,j) != cell(i,j)` (two
  segments both "up 65%" → accidental truth → discard).
- **NEG-col:** cell (i,j) re-attributed to header j′ ("operating income" value claimed as "gross
  margin"). Guard `cell(i,j′) != cell(i,j)`.
- **NEG-aggregate-bind — mint deliberately, this IS ar-12:** a value stated in PROSE bound to the
  document/consolidated subject within the chunk, rebound to a co-present table row label — and the
  symmetric twin (row value → consolidated subject). The ar-12 chunk (BUSINESS OVERVIEW + segment
  table, verified in `scope_probe_fp.json`) is the template; mint from NON-holdout docs (other
  10-K sections, CCNA decks, linux PDFs share the shape). Target ≥1.5K of these.
- **Total-row guard:** never rebind onto "Total"/"Ensemble" rows without the F2 recompute check
  (reuse the verify sum-expr backstop pattern).
- **FR tables:** row labels are proper-ish tech nouns (OSPF, VLAN) → deterministic splice safe;
  FR sentence negatives go through mask-refill (gender/agreement disfluency tell avoidance).

### 2.4 Scale + class balance
- **Target 12K rows (6K OK / 6K BREACH), 1:1 globally AND per source-type.** Mix: ~40% table-minted
  (incl. the ≥1.5K aggregate-bind class), ~40% prose entity-rebind, ~10–15% French, ~10%
  unit-transform/format-diversity positives. **Ladder: train at 4K first; extend toward 12K only if
  dev F1 hasn't saturated.**
- Combined training set = mints + RAGTruth-EN + ragtruth-fr-translated replay (replay ratio is an
  open knob — start 1:1 mints:replay, mini-sweep on the MINTED dev split only; R1 open-q #1).
- Generation budget: ~2–4K chunks × (1 extract + ~3 paraphrase) guided calls on the daemon ≈ a few
  hours batched. HYPOTHESIS on exact wall-clock; measure on the first 200 chunks.

### 2.5 Quality filters (F1–F7)
- **F1 duplicate-value guard (deterministic):** value occurs >1× in chunk → resolve each
  occurrence's binding (table lookup / sentence-subject match); any occurrence binds to S2 → discard.
- **F2 table truth-check (deterministic):** direct cell lookup at (S2_row, predicate_col); equal →
  discard; sum-shaped values recomputed.
- **F3 coreference/alias guard:** substring, initialism (`index/initialism.py`), "the company"≈org.
- **F4 NLI accidental-truth discard:** DeBERTa-MNLI (cached), premise=chunk, entailment>0.5 →
  discard. Direction-correct per audit-18; fine as a high-precision discard filter though dead as a
  checker.
- **F5 minimal-pair integrity:** byte-diff covers exactly the subject span (post-refill assert).
- **F6 fluency (optional):** 4B logprob to drop egregious splices; never to label.
- **F7 caps:** ≤6/chunk, ≤400/doc, ≤15% per template; both classes pass through the paraphraser at
  the same rate (style symmetry); positives minted by the same extract/template machinery as
  negatives (edit-profile symmetry).

### 2.6 The calibration-set-is-never-training-data rule (hard)
1. **Doc-level exclusion:** any doc cited by any of the 14 calibration tuples is excluded from
   minting entirely.
2. **Automated leak assert in the mint script (CI-style hard fail):** no minted chunk_id ∈
   calibration chunk_ids; no minted claim with token-Jaccard >0.8 vs any calibration claim/summary.
3. **Dev split BY DOCUMENT, never by row** (random row splits leak chunk context).
4. The calibration set influences **nothing**: not training, not thresholding, not early stopping,
   not the replay-ratio sweep. It is evaluated once per shipped candidate (§4).

---

## 3. Training plan

**Script shape — in-repo deps ONLY, no new extra:** `scripts/train_binding_checker.py` +
`scripts/mint_binding_data.py`, vendoring (MIT, with attribution headers) the ~300 lines of
`lettucedetect`'s `Trainer` + `HallucinationDataset` + evaluator from the cached wheel. transformers
4.57.6 + accelerate 1.13.0 are already locked (HF `Trainer` available); torch 2.11.0. **Full
fine-tune, no LoRA** (`peft` is not in the lock — full-FT of ≤307M makes the new-extra question
moot). **Plain `torch.utils.data.Dataset`** (the locked `datasets` 2.14.4 is old/ragas-pinned —
avoid it). Data files = LettuceDetect `HallucinationSample` JSON.

**Hyperparams (TinyLettuce/LettuceDetect lineage):** AdamW lr 1e-5, weight decay 0.01, no
scheduler/warmup; epochs 3–6 (select-best per epoch); batch 8 (grad-accum or gradient-checkpointing
if seq-4096 @ 307M overflows; drop to bs 4 before dropping seq len); `max_length=4096`,
`truncation="only_first"` (prompt side truncates, answer labels always survive);
`DataCollatorForTokenClassification(label_pad_token_id=-100)`; **checkpoint selection = best
class-1 token F1 on the MINTED dev split** (per the upstream `trainer.py:108-112` discipline).

**Where it runs:** GPU with the daemon paused — wrap the train invocation in
`pause_vllm_for_gpu()` (`parse/pipeline.py:875`; zero-arg asynccontextmanager, no-op if vLLM down,
bounded-retry restart in `finally`). Sequencing: **mint with the daemon UP** (4B guided calls) →
**train with the daemon PAUSED**. Fallback: ettin-encoder-68m arm trains on CPU overnight if GPU
contention bites.

**Pre-training probes (P0, blocking):**
1. **mmBERT tokenizer-offset probe (~15 min):** pair-encode 20 synthetic examples, assert the
   `answer_start_token` heuristic + offset mapping label the intended answer chars (Gemma-2
   tokenizer is unproven in this recipe — R1 open-q #4). If it fails and isn't trivially fixable →
   fall back to the EN continued-FT arm + accept a known-FR-gap, or fix the heuristic.
2. **Span-target probe (cheap A/B inside the first 4K run):** label only the swapped-subject span
   vs the whole claim sentence (R1 open-q #3); pick by minted-dev F1.

**Wall-clock estimate (HYPOTHESIS — no published number anywhere; measure, don't assert):**
307M @ seq 4096, ~10–30K rows × 3–6 epochs on the 4070 ≈ **1–4 h**; ettin-68m arm minutes-to-1h
GPU / overnight CPU. Record the measured figure in the audit.

**Candidate ladder (train in this order, stop at first gate-PASS):**
1. mmBERT-base on mints + RAGTruth-EN + ragtruth-fr-translated (primary, FR+EN single model).
2. `lettucedect-base-modernbert-en-v1` continued-FT on mints + RAGTruth replay (EN-fastest,
   inherits the 9/12-clean FP profile; FR gap accepted only if arm 1 fails).
3. `ettin-encoder-68m` synthetic-heavy (cheapest, CPU-deployable inference).

---

## 4. Gate protocol on the frozen calibration set

**Harness:** a new `binding` subcommand in `scripts/scope_guard_span_probe.py` (8th arm; exact
precedent of the 7 existing). Consumes `_checker_cases(fp_path)` (line 408: BREACH+FP cases with
`premise_raw`/`premise_lin`/`sentences`) — i.e. zero new case-building code; scores each case with
the fine-tuned checkpoint via the existing `lettuce_arm` inference shape (same `_LD_QA_TEMPLATE`,
same `ld_max_conf` scalar, question/no-question/tail-stripped variants); emits
`{side, qid, bc_max_conf_*}` rows → `_checker_report` (margin = `min(BREACH) − max(FP)`,
SEPARATES/OVERLAP verdict) → `scope_probe_binding.json` alongside the prior arms' artifacts.

**Pre-req hardening (do before the first gate run):** freeze the two breach chunk TEXTS into the
calibration artifact (e.g. `docs/audits/data-17-scope-calibration/breach_chunks_frozen.json`).
Today they are live-fetched by suffix from `search.sqlite` (`fetch_chunks_by_suffix`, FATAL on
miss) — a reindex churns chunk_ids (the chart-types 06-01 lesson; R3 caution #2). The harness
prefers the frozen texts, falls back to live fetch with a loud warning.

**Threshold-selection discipline (hard rule):** the operating threshold is chosen ONCE, on the
**minted dev split** (by-document holdout) — e.g. the threshold maximizing dev F1 or fixing dev-FP
≤2% — and frozen into the candidate's metadata BEFORE the calibration run. **The calibration set is
a one-shot GO/NO-GO gate per shipped candidate: it never tunes the threshold, never selects the
checkpoint, never feeds back into training.** No post-hoc threshold movement to convert a FAIL into
a PASS — that is the whack-a-mole anti-pattern (CLAUDE.md measure-don't-assert #4).

**PASS bar (all three, at the pre-frozen threshold):**
1. **Both breaches caught:** ar-12 AND tg-13 score ≥ threshold. (tg-13 is already closed in
   production by the deterministic provenance backstop — ar-12 is the marginal-value exemplar — but
   the gate keeps both: a binding checker that misses tg-13's shape is suspect.)
2. **FP budget: 0 of the 12 FPs fire** (12/12 clear). If exactly one fires, the increment is a
   NO-GO unless re-adjudication shows that tuple's gold is wrong (gold correction is allowed; gate
   re-tuning is not).
3. **Positive margin:** `min(BREACH) − max(FP) > 0`, reported; a knife-edge pass (margin < ~0.05)
   ships only with the next-rung validation anyway (below) and is flagged in the audit.

**Determinism:** score N=2 runs; the encoder forward is deterministic on fixed hardware so rows
must be byte-stable — any instability is a harness bug, fix before judging.

**The calibration gate is necessary, not sufficient.** Wiring touches the answer path → per
CLAUDE.md measure-don't-assert #3/#4 the SHIP gate is the REAL full ladder: `memex eval` across all
corpora, multi-run N≥2–3 (the `raw/final_ladder` pattern), refusal_cf must hold 1.0, ANS counts no
worse than baseline (an over-refusal regression from checker FPs is a first-class failure).

---

## 5. Wiring plan

**Where it sits:** mirror of the provenance-backstop pattern, as a second advisory check at the top
of `assess_relevance` (`agents/answering.py:2429`), immediately after `_provenance_scope_violation`
(line 2351). New `_binding_violation(state) -> str | None`:
- Inputs (all already on `AnswerState` at that point): `state.query`, `state.draft.summary`,
  grounded claims via the `state.draft.claims` × `state.verification.grounded` join (pattern at
  answering.py:2380-2388), cited chunk title/text from the `state.reranked` window.
- Build the `_LD_QA_TEMPLATE` prompt (cited chunks as passages) + the summary as the answer; one
  forward; fire iff `max class-1 span conf ≥ frozen threshold`.
- Fire → `RelevanceAssessment(responsive=False, reason=...)` → routes to `refuse`. **Advisory: can
  only narrow, never admit** — it can never inject content or un-refuse.

**Fail-open semantics (exact precedents):** model-load or inference error → log
`binding_checker.failopen` and return None (the provenance backstop's store-failopen shape,
answering.py:2400-2410; the relevance node's ModelCallError→default-responsive shape,
answering.py:2492-2504). A checker failure must never manufacture a refusal. Config read under
`ConfigurationError` → disabled.

**Kill-switch:** `AgentsSettings.binding_checker_enabled: bool = False`
(`core/config.py`, beside `provenance_scope_enabled:710`) → env
`MEMEX_AGENTS__BINDING_CHECKER_ENABLED`. Plus `binding_checker_threshold: float` (the frozen value)
and `binding_checker_model_path: str`. **Ship default-OFF; flip to default-ON only after the full
ladder passes N≥2–3** (the provenance backstop's own ship sequence).

**Model lifecycle:** the OTTER pattern (`enrich/ner_otter.py` — lazy process-global, lock-serialized
forward, CPU default), NOT the model registry: an advisory checker must not couple into the
auto-mode VRAM floor or OOM-breaker. **CPU inference, fp32, ~0.6–1.2 GB RAM** for 149–307M — zero
VRAM budget, preserves eval determinism (the "eval must pin a device" rule).

**Latency budget:** one ≤4096-token encoder forward per /ask on CPU. **HYPOTHESIS: O(0.5–3 s)**
(the bge CPU-rerank precedent suggests seconds-scale for ~300M @ long seq) — measure on the
calibration tuples; if >3 s it still rides inside the existing ~10–60 s answer latency, but record
it. GPU placement is an optional later knob, not v1.

---

## 6. RISKS + kill criteria + the pre-training probe

### Cheap probe BEFORE training (do first)
- **`osunlp/attrscore-flan-t5-large` (~780M, apache-2.0; R4's only candidate):** ~30 min — one
  pre-air-gap download, run through a one-off arm against the frozen calibration set. The only
  off-the-shelf checkpoint with any attribution-error training (3-way
  Attributable/Contradictory/Extrapolatory). Expected to fail (no GFM exposure, EN-only —
  HYPOTHESIS, untested); it is the last box to tick. **If it unexpectedly PASSES the §4 bar, wire
  it and skip the fine-tune entirely.**
- TAPAS/TAPEX-TabFact: probe ONLY if ar-12's 71.1% sits inside the parsed table (it sits in prose
  per R2's reading of the chunk — so almost certainly skip; the input-format mismatch is structural).
- The two P0 probes from §3 (tokenizer offsets; rag-fact-checker guided-JSON against local vLLM if
  the LLM-perturber supplement is used — 5 min) run before any GPU spend.

### Risks (each with its mitigation)
1. **Shortcut learning** — style leak / inverse-presence / verbatim-value / template memorization /
   edit-distance artifacts → §2.5 stages C+D + A3 + F7 symmetry; diagnose via the question-stripped
   and tail-stripped scoring variants (audit-18's style-noise detector).
2. **Accidentally-true negatives = label noise** → F1/F2/F4 (the AMRFact NegFilter analog is a
   named, load-bearing component).
3. **mmBERT tokenizer-offset failure** → P0 probe; fallback arm 2.
4. **FR disfluency tell in splices** → all FR negatives via mask-refill + F5 byte-diff assert.
5. **Stale-vault chunk drift invalidates the gate** → freeze breach chunk texts (§4 pre-req).
6. **Checker FPs cause over-refusals on the full ladder** (the first-class failure) → default-OFF
   flag, full-ladder N≥2–3 before flipping, fail-open everywhere.
7. **Calibration overfit by iteration** — repeated gate runs leak information through the operator →
   the candidate ladder is bounded (≤3 trained candidates, §3); each gets ONE gate run.
8. **4B minting quality** (extractor misses subjects, refill edits beyond the span) → F5 hard
   assert discards; extraction validated on a 50-row manual audit before scaling (measure, don't
   assert).
9. **Wheel-vendoring drift** → vendored files pinned to lettucedetect 0.1.8 with source-path
   comments; no runtime dep on the pip package.

### Kill criteria (each recorded as a first-class negative: `docs/audits/NN-binding-checker.md` +
memory topic + ROADMAP "tried + reverted", with a do-not-re-walk note)
- **K1 (data kill):** minted-dev class-1 F1 < ~0.75 at 12K rows after the span-target A/B — the
  class isn't learnable at this scale/architecture; stop before burning more candidates.
- **K2 (gate kill):** all ladder candidates (≤3) fail the §4 bar under threshold discipline —
  record per-candidate margins; the banked verdict reverts to "content class = residual, deterministic
  backstops only"; do NOT re-tune thresholds against the calibration set to rescue a candidate.
- **K3 (ladder kill):** calibration PASS but the full `memex eval` ladder shows refusal_cf < 1.0 or
  a net-ANS regression at N≥2–3 with the flag ON → ship default-OFF (opt-in like
  `usage_intent_demotion_enabled`) or revert; the calibration set then gains the regressing shapes
  as named residuals.
- **K4 (probe windfall):** attrscore passes §4 → the fine-tune is killed by success; wire the
  off-the-shelf model instead (cheap probes before expensive work).

### Increment order (buildable sequence)
1. Freeze breach chunk texts into the calibration artifact. 2. attrscore 30-min probe. 3. mmBERT
tokenizer P0 probe. 4. `scripts/mint_binding_data.py` (mint 4K + leak asserts + 50-row manual
audit). 5. Vendor trainer; train candidate 1 under `pause_vllm_for_gpu`; freeze threshold on minted
dev. 6. One-shot calibration gate via the new `binding` probe subcommand. 7. (PASS) wire
`_binding_violation` default-OFF → full ladder N≥2–3 → flip default-ON. 8. (FAIL) ladder to
candidate 2/3 or record the negative per K1–K3.

---

## 7. Build log (2026-06-12/13, branch `feat/binding-checker`)

**Kill-target re-confirmed on shipped main `be0a209`:** ar-12 breaches 3/3 under the
mxbai env with the provenance backstop ON (correctly silent — content subject, not a
named source). The verified CLAIM is clean ("Gross margin was 71.1%…"); the SUMMARY
carries the binding — confirming the checker gates the final surfaced text.

**attrscore (the §6 windfall probe): DEAD — the 8th measured arm.** Both breaches
labeled Attributable (qa_raw 0.460/0.614); 7/12 FPs at-or-below the top breach on
every variant (margin −0.401/−0.495). Same overlap-bias family as HHEM. Verbatim
template from the AttrScore repo; rows in `data-17-scope-calibration/scope_probe_attrscore.json`.

**P0 tokenizer probe: mmBERT GO.** 7/8 identical to the known-working lettucedect-en
control (FR accents, GFM, span boundaries all clean). The shared 8th case exposed an
UPSTREAM recipe bug: `answer_start_token = len(encode(context alone))` loses the
answer entirely when the context alone exceeds max_length. Vendored fix: locate the
answer region via fast-tokenizer `sequence_ids()` — exact on both tokenizers under
truncation. Pinned by `tests/unit/test_binding_checker_vendor.py`.

**Pre-req hardening:** breach chunk texts frozen into
`data-17-scope-calibration/breach_chunks_frozen.json` (reindex churns chunk ids — the
chart-types 06-01 lesson); `_checker_cases` prefers frozen, live fetch is the loud
fallback; `question` threaded through the cases for claim-format checkers.

**Minter (`scripts/mint_binding_data.py`) deltas vs the §2 spec, all measured-in:**
- Numeric-metric tables are SCARCE outside the held-out 10-K (vault probe: 243
  rebindable tables, mostly CCNA/Linux command-option tables; 13 tables in
  gte/chart/nist of which 9 numeric-majority) → numeric-first cell ordering + a 2×
  per-chunk cap for numeric tables; value-typed template banks (metric verbs only on
  numeric cells — a "Long Option reached COMMENT" tell would leak class signal).
- THE DAEMON SILENTLY IGNORES vLLM's bare `guided_json` (deprecated) — extraction came
  back fenced/free-form. Fixed to the OpenAI-standard
  `response_format={"type":"json_schema"}` (what production `models/client.py` uses).
- Subjects constrained to the chunk's OTTER entities ∪ table row labels, passed INTO
  the extraction prompt — kills generic-NP subjects ("Examples of fuzzers") and
  guarantees the rebind pool is kind-matched; a per-language verb gate drops
  title-shaped extractions.
- F4 NLI discard: free-VRAM pre-check (≥4 GB) + per-batch OOM→CPU fallback (first full
  run died OOM beside the 6 GB daemon — forward was unguarded); phase-P raw samples
  now saved pre-F4 so a crash can't lose LLM work.
- Phase-T manual audit (the §2 50-row check): rebinds are genuinely
  presence-preserving (PF-as-fuzzer, category rebinds, key-value swaps); spans land on
  the swapped slot; hard positives present.

**Gate harness:** `scope_guard_span_probe.py binding <fp> <out> <model_dir>` — scores
the frozen 14 cases via the UNCHANGED lettuce_arm machinery (train == gate == wiring
input shape), asserts N=2 byte-stability, applies the §4 bar at the threshold frozen
in `<model_dir>/threshold.json`: both breaches ≥ t on conf_q AND surviving tail-strip
(conf_qs ≥ t), 0/12 FPs ≥ t, margin reported with a knife-edge flag.

## 8. The candidate ladder (one gate run each — §6 risk 7 discipline)

**Candidate 1 — mmBERT-base fresh, mints v1 + replay: GATE FAIL (2026-06-13).**
Dev was excellent (token P=1.000/R=0.707/F1=0.829; example-F1 0.870 at t=0.5, dev FP
0.000; the K1 learnability bar CLEARED — the binding class IS learnable from minimal
pairs). The calibration gate inverted it: ar-12 MISSED at 0.0 (tg-13 caught 0.704),
6/12 FPs FIRED at 0.70–0.95. Span autopsy (the audit-18 instrument lesson): every FP
fire sits on PROVENANCE-TAIL/doc-name tokens ("2026 Annual Review (Form 10-K",
"SP 800-207", " Linux") — the documented audit-18 citation-tail FP mode, which §2.5
spec'd as a REQUIRED positive class and mint v1 under-delivered (~0 tail-style rows,
2 unit-transform rows); ar-12's metric-possessive production shape ("the gross margin
for X … was V, as stated in …") was likewise absent (the §2.3 deliberate
aggregate-bind class effectively didn't materialize). Verdict: a TRAIN/PRODUCTION
style gap — an implementation-vs-spec shortfall, not a learnability kill. Rows:
`data-17-scope-calibration/scope_probe_binding_cand1.json`.

**Mint v2 (completing the §2.3/§2.5 spec):** provenance-tail style symmetry (", as
stated in the {doc_title}." appended to ~35% of BOTH classes, 814 rows — tails must
be a POSITIVE style, never a breach signal); possessive metric templates ("{label}'s
{header} was {value}"); the deliberate aggregate-bind + key-value sibling-rebind
miners. **Corpus fact (honest):** the financial metric-prose register ("X was 71.1%")
exists in ZERO non-holdout chunks — it lives in the held-out 10-K; the vault's real
binding shapes are "is N" technicals and "K: V" lines, so the agg/kv classes yield
only ~33 rows and the v2 lever is tails + possessive templates. 2,299 samples
(803 breach / 1,496 ok), prose phase reused from v1 (deterministic re-mint, no LLM).

**Candidate 2 — lettucedect-base-modernbert-en-v1 continued-FT on mints v2 + replay**
(the pre-declared arm 2; inherits the RAGTruth FP discipline; EN base accepted with
the FR gap noted): training 2026-06-13.
