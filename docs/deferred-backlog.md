# Deferred backlog — items not yet implemented

**Generated:** 2026-06-05 · **Last reconciled:** 2026-06-13 · **Pending items:** ~123 (+ 40 explicitly decided-against).

> **Reconciled 2026-06-13 — answer-TEXT-correctness eval extended to ALL 13 prose corpora; the 2 catches it surfaced ROOT-CAUSED → DEFER (do-not-re-walk).** `answer_must_mention`→`answer_mention_recall` now annotates every prose answer-eval corpus (~184 slots; nist `210c82b` + clean-4 `e086891` + noisier-8 `f565f77`; refusal_cf=1.0, N=2 byte-stable). The metric is a DIAGNOSTIC — the answer STAGE is sound (isolated probe: the 4B counts 7 GIVEN the full list). Its 2 catches are UPSTREAM, both DEFERRED as root-caused residuals (workflow-verified `wb6zr4qwm`): **(1) nist-01** "how many tenets" answers 4 not 7 — the 7-tenet list spans 4 chunks / 5564 chars and cannot co-reside in the top-5 × 1800-char window inside the 8192-token 4B; every lever fails (chunk-atomic infeasible; k=8 reverted net −3 / k=18 overflows→refuses; an answer-prompt count-nudge collides with v5 literal-presence + the verify numeric backstop = the ADR-0022 v6 trap; section-aware retrieval backfires). Clean future lever = a per-query adaptive window / sharper reranker / larger orchestrator window. **(2) chart-types-09** lists 5/6 GIF assignees — a chart-OCR continuation-row table the LLM mis-reads; a `table_linearize` forward-fill is UNSAFE (corrupts legit non-continuation empties across 18 docs/49 chunks) + ineffective (table is header-gate-skipped → raw GFM) + churns chunk_ids. **SIDE-FINDING → FIXED 2026-06-13: stale eval gold across 5 corpora RE-ANCHORED.** The audit found **35 stale gold `relevant_chunk_ids` across 5 corpora** (annual-report, scientific-gte, nist, forms-w9, chart-types; the other 9 clean) — the docs were re-chunked since the gold was anchored, deflating the ladder cp toward 0. All re-anchored via FTS resolution of `_anchor_phrase` (chart-content anchors are FTS-stripped → manually anchored to the chart chunk); 0 stale remain; **cp_answered recovered: nist 0.0→0.908 / forms-w9 0.917 / chart-types 0.857 / scientific-gte 0.732 / annual-report 0.350** (the last is the pre-existing SIBLING-chunk artifact — the 10-K repeats each figure across chunks, so single-chunk gold can't match a sibling citation — NOT staleness; multi-chunk gold would lift it but is cp-tuning). refusal_cf=1.0 + answer_text byte-stable (gold is citation-only ⇒ HARD-gate-neutral). LESSON: a re-parse/re-chunk SILENTLY staleness the content-addressed gold chunk_ids → audit + re-anchor after any re-index. The FR language-drift (tg-07/ccna-02) stays a soft UX observation.

> **Reconciled 2026-06-13 — the sharper-reranker (mxbai) A/B + summary-scope arc TERMINALLY CLOSED (audits 17/18/19).** The "clean fix is a SHARPER RERANKER" lever banked under the codebase-corpus item below was measured and is now BLOCKED: `mxbai-rerank-base-v2` sweeps the rank case-files 10/10 but makes a pre-existing summary-scope hole deterministically reachable. The hole split → **provenance-class SHIPPED** (deterministic provenance-scope backstop, default-ON, `MEMEX_AGENTS__PROVENANCE_SCOPE_ENABLED`, audit-18) + **content-class (ar-12 binding fabrication) = documented residual**, BOTH cheap levers tried+reverted (gate-detection fine-tune = K2 structural negative, audit-19 §9; `answer@v6` generation prompt = reverted for blast radius, §10). Terminal: mxbai stays blocked; content-class production exposure UNCHANGED (mxbai-reachable only, the shipped bge HARD gate holds); the only remaining instrument is far-future complete-evidence-entailment / masked-subject machinery (NOT queued — reopen ONLY if a future reranker swap makes the breach production-reachable). The **answer-TEXT-correctness eval** (the measure-first prerequisite, already shipped for codex-rs as `answer_mention_recall`) is now the actionable autonomous successor lever — extend it to the prose corpora before any answer-stage work.

> **Reconciled 2026-06-09 — codebase-corpus + false-refusal arcs shipped (ADR-0021/0022).** Added two 🎯-Queued items: **codebase corpus Phases 3–5** (BM25-for-code measure-first + the full codex-rs corpus + `gold_chunk_recall@k`; Phases 1–2 merged `268e39e`/`6e477b3`/`b9c3a58`) and the **18 residual false-refusal classes** (retrieval-miss / unreadable-figure / synthesis — NOT advisory-gate-fixable; ADR-0022 cut 30→18). The doc-sync checkpoint also refreshed ROADMAP/ADRs/CLAUDE.md.

> **Reconciled 2026-06-07 — autonomous-actionability audit (workflow `wf_76f65abf`).** Verified the top autonomously-actionable candidates against live code. **Finding: the autonomously-actionable well is largely dry** — every higher-value candidate is curator-/corpus-/hardware-gated or a measured defer. Resolved: the 3 "Uncertain" items are ALL shipped/subsumed (section below); **de-hyphenation = measured DEFER** (0 `word-\nword` occurrences corpus-wide); the stale `registry.py` "Qwen3-8B" docstring FIXED → the 4B. The lone clean S-pickup left is the doc-picker "Clear selection" control (low value); its pre-emptive count-badge half is M-effort (needs an OOB picker re-render on `/ask`, can't see manual ticks otherwise).

> **What this is.** A point-in-time snapshot of every consciously-deferred item across the project,
> synthesized from a sweep of `docs/ROADMAP.md`, the ADRs, the specs, the agent's memory trackers,
> the audit reports, and code comments — deduplicated, with anything since-shipped removed. It is a
> **regenerable digest, not live state**: an item here may have shipped since; verify against current
> code/ROADMAP before treating any line as fact. The authoritative trackers remain ROADMAP.md (status)
> and the per-feature ADRs/specs (the *why*). Many entries are granular — some are sub-items of a larger arc.

> **Reconciled 2026-06-05:** dropped the orphaned `duckdb` dep (pruned) and the entire
> "audit-2026-05-20 open items" entry — all four sub-items (D2 `_DOC_LOCKS`, #23 vendored Tailwind+HTMX,
> #27 watcher test, #29 MCP returns pydantic) verified DONE.
>
> **Reconciled 2026-06-06 — the four "queued" items knocked out:** (3) **eval-expert MAJORITY-of-N** SHIPPED
> (`fa1342d`, eval-only). (4) **UI-ingestion residuals** — verified ~90% already shipped (chunk_count gate,
> half-doc detect, chart-OCR unload, orchestrator reconcile); only half-doc resume/sweep remains, deferred-
> by-design (auto-acting at startup is unsafe; the manual `memex index <doc>` path exists). (1) **VLM W6**
> — the V2 prompt (decoration-skip WITHOUT the verbosity-inducing preservation guard) SHIPPED (`c911a80`),
> validated on the regression site (cr350-diagrams 11/11 ANS, no −1); the **vault-wide re-parse migration
> RAN 2026-06-06 as a STAGED PARTIAL → DONE** (`w6-migration-2026-06-06`): the **22 eval-covered VLM docs
> re-parsed to V2 + KEPT**, the **25 unvalidatable HELD on baseline** (deliberate MIXED vault). HARD gate held
> (refusal_cf=1.0, 0 halluc, ccna +1); 3 stable −1 ANS (2 recoverable gate over-refusals + 1 retrieval miss).
> **LESSON: re-transcribing a non-deterministic VLM doc churns chunks → retrieval/grounding shift even with
> NO content loss.** (2) **Docling re-tiering** — premise confirmed via diagnostic but it's a redesign, not a
> tweak (see Queued).
>
> **Follow-ups from the W6 migration:** gold-anchor re-baseline for the 7 affected corpora ✅ DONE 2026-06-06
> (see "deferred-but-since-shipped"); the grounding-gate over-refusal investigation ✅; the slide-decks
> verify-JSON-overflow eval-robustness gap (STILL OPEN — re-confirmed by the re-baseline; see the eval-runner item).

## How to read the categories

- **🎯 Queued / next-pickup** — intended soon.
- **⛔ Data-gated / blocked** — needs corpus data, a model/hardware, or corpus scale before it can even be measured.
- **📋 Feature backlog** — would-do, lower priority, no external blocker.
- **🔬 Researched + banked** — has a verdict; waiting on a forcing function.
- **🚫 Decided-against** — tried-and-reverted or rejected; **NOT pending** — listed only so they're not mistaken for open work.

## 🎯 Queued / next-pickup (3)

_Intended soon — the most actionable._

- **✅ Codebase corpus Phases 3–5 — SHIPPED + merged `f0f47bb` (2026-06-09).** Phase-3 BM25-for-code "Lever A" default-ON (`8b325db`, audit-13) + Phases 4-5 full codex-rs (99 `.rs`) as a permanent MAIN-vault corpus + a 47-query find-the-code baseline (audit-14). The ADR-0021 arc is CLOSED. **Genuinely-banked sub-items remain:** (a) a **Python** symbol splitter (a separate increment — `# comment` / significant-whitespace misfires; Rust-only in v1); (b) in-source `#fn-foo` jump-to-symbol citation anchors (v2); (c) the **usage-class answer-stage lever** — the rerank-demotion remedy was measured DOUBLE-EDGED (+3/−2, audit-14) and shipped DEFAULT-OFF; ~~the clean fix is a SHARPER RERANKER~~ **the sharper-reranker candidate (`mxbai-rerank-base-v2`) was MEASURED and BLOCKED 2026-06-11/12/13 (audits 17/18/19 — it opens a summary-scope hole; see the 2026-06-13 reconciliation note above)**, so the actionable lever is now the **answer-TEXT-correctness eval** (measure-first; already shipped for codex-rs as `answer_mention_recall` — `memex eval` scores answered/cp, which HID this; extend it to prose). Do NOT reopen as rerank-heuristic tuning (whack-a-mole — no clean rule) and do NOT re-attempt mxbai without first closing the content-class scope hole.  
  *Sources:* docs/specs/code-chunking.md; docs/adr/0021-codebase-corpus-code-as-documents.md; index/rust_symbols.py; the `codebase-corpus-design-2026-06-07` memory.
- **False-refusal residual classes (18 remaining after audit-12) — DEFERRED, recorded (NOT advisory-gate-fixable).** The advisory-gate rebalance (ADR-0022, `0d76ace`) cut false refusals 30→18; the residual 18 are not a prompt fix: ~**retrieval-miss** (gold never reranked — a retrieval/embedder/recall lever, the P2.5 embedder-swap territory), **unreadable-figure-data** (slide-decks-04/16 — the answer sits in a figure the VLM didn't transcribe legibly), and **genuine partial-topic / SYNTHESIS-class** (the evidence is co-located in one chunk but needs a one-step INFERENCE the chunk never states, e.g. cr350-img-01 — the reason-over-evidence direction, the SAME as the reverted contextual-retrieval experiment that broke the HARD gate). All HARD-gate-safe (a false refusal is never a fabrication); fixing them needs retrieval/reasoning work, not a gate loosen.  
  *Sources:* docs/audits/12-false-refusals.md; docs/adr/0022-advisory-gate-rebalance-false-refusals.md; src/memex/CLAUDE.md (the assess / answer-prompt / relevance-gate bullets).

> **✅ DONE 2026-06-06 — Daemon-state silent-404 after a CLI VLM/chart-OCR parse (ROOT-FIX, HIGH).**
> A `memex parse`/`index`/`reindex` VLM/chart-OCR pass left the orchestrator serving the 8B (→ `/ask`
> 404s) + an un-killable stray. Root cause = THREE stacked spawn bugs in `parse/pipeline.py::_vllm_restart`
> (no serve-env → 8B; `nohup` → dead `uv run` leader → orphaned vLLM; `asyncio.create_subprocess_exec` →
> loop-close SIGKILL → same orphan). Fix: hoisted `orchestrator_serve_env`+PID helpers to
> `core/model_serving.py`; `_vllm_restart` now injects the serve-env + writes the PID file + spawns via
> SYNC `subprocess.Popen` (mirrors `start()`). Live-validated (`doctor orchestrator_match=True`, clean
> `daemon restart`). `stop()` left untouched (its leader-only wait is pre-existing + out of scope).
> See `daemon_state_silent_404_2026_06_06` + ADR-0015 Amendment (2026-06-06).

- **Docling mode-anchored re-tiering on dense UNNUMBERED docs (step-3b residual) — INVESTIGATED 2026-06-06 → RE-RUNNABLE DEFER (no demonstrated beneficiary; NOT an impossibility proof).** Goal: un-flatten a buried dominant section tier on a force-docling'd dense unnumbered doc so its real titles aren't pinned at the H5 cap. A **full-distribution diagnostic** (`scripts/docling_heading_histogram.py`, COMMITTED; reuses the production `docling_worker.bucketed_header_heights` seam so it measures exactly what `_recover_heading_levels` sees) captured the real per-bucket height→frequency on the 3 force-docling'd candidates. **The premise was REFINED, not confirmed:** the "annual-report" 10-K is a 3-DOCUMENT concatenation (proxy / annual-review / 10-K) with genuinely MULTI-MODAL real tiers — post-prose-demotion: mode **8.0pt×271** (proxy sections) sits UNDER populous real tiers **9.0×80 / 10.0×24 / 19.0×28** (review + risk-factor titles) + 7.0×27 (board names) + 65–102pt banners — NOT a single dominant tier under a thin scatter of rare singletons. The designed safe mode-anchor (a dominance gate + a **PYRAMID-INVERSION guard**) correctly **DECLINES to fire on all 3 docs**: 10-K (populous tiers ABOVE the mode → firing would over-flatten the banner/risk-factor tiers); CUDA deck (varied-height, NOT uniform → already spread by size-primary); NIST (NUMBERED → the downstream normalizer already fixes it). So **no available doc fits the clean buried-dominant-tier model to even exercise a successful fire** → shipping the heuristic would be unvalidated + inert. Size-primary `_recover_heading_levels` left UNCHANGED. Marginal value reconfirmed (the vault 10-K's DEFAULT route is the clean PyMuPDF tree).  
  *Unblock (re-runnable):* if a real force-docling'd dense UNNUMBERED doc with a clean dominant-tier-under-rare-scatter shape appears, re-run `uv run python scripts/docling_heading_histogram.py <doc> --post-demote`; build the gate only if the histogram shows a SEPARABLE dominant tier. See [[docling-retier-nogo-2026-06-06]].  
  *Sources:* scripts/docling_heading_histogram.py; docling_worker.py::bucketed_header_heights / _recover_heading_levels; audits/10:169

## ⛔ Data-gated / blocked (38)

_Would do, but blocked on curator corpus data, a model/hardware availability, or large-corpus scale._

- **P0 corpus — curator-gated REAL documents (3 untouched categories)** — Hand-curated REAL PDFs per category + the 3 untouched categories (modern-printed REAL, historical-scans, handwritten); current cross-corpus numbers use synthetic fixtures.  
  *Unblock:* Curator time + source material; the repeatable add-a-corpus playbook is proven. The single most-impactful pending item. Scans+handwriting also unblock further VLM exercise.  
  *Sources:* ROADMAP.md:54/208/391/507; next_priorities.md:67-70; p0_corpus_nist; build_status.md
- **P0 parse-eval — hand-curated ground-truth.md per real doc (CER/WER/structural-F1 at scale)** — Run CER/WER + heading/table/equation structural-F1 against hand-transcribed reference markdown at scale (only 3 synthetic fixtures exist).  
  *Unblock:* Plumbing (run_parse_eval / memex eval-parse) is live; needs curator work — a hand-transcribed ground-truth.md per doc.  
  *Sources:* ROADMAP.md:211/508
- **P0 chart-DOMINANT corpora (financial dashboards, infographic decks)** — Chart-heavy corpora to exercise the post-v7 chart-content answering stack.  
  *Unblock:* Curator-gated: needs chart-dominant source docs to validate v7 on additional chart-dense material.  
  *Sources:* ROADMAP.md:391/395
- **Expand handwritten + historical-scans eval corpora** — Widen the scan->VLM/handwritten corpora with curated real pages (CS-Notes/Muharaf/GNHK/IAM/historical) + authored queries (scan route + cs-notes-1 already shipped).  
  *Unblock:* Curator-gated (not autonomously executable): needs the user to pick pages + author should-answer/counterfactual queries.  
  *Sources:* next_priorities.md:68; handwritten_corpus_research_2026_05_27
- **P2.1 — Qwen3-Reranker-0.6B quality A/B per-category re-run at corpus scale** — Re-run the reranker backend A/B (qwen3 vs cross_encoder) per category on a deeper corpus. Infra + a 1-doc verdict shipped (cross_encoder won, qwen3 stays opt-in); the scaled per-category verdict has NOT run.  
  *Unblock:* Needs P0 corpus depth to discriminate between candidates (Tier 3).  
  *Sources:* ROADMAP.md:49/221/272/611; reranker_gpu_ab_2026_06_01; build_status.md (ADR-0001 reranker candidate)
- **P2.2 — Granite 4.1-8B-FP8 vs Qwen orchestrator A/B** — Alternative-orchestrator A/B against Granite 4.1-8B (hybrid Mamba2+attention).  
  *Unblock:* vLLM-blocked: vLLM 0.21 hangs at FA2 init for Granite's hybrid arch. Unblocks on vLLM 0.22+ hybrid-arch support, a Granite GGUF variant, or a Granite 3.x fallback.  
  *Sources:* ROADMAP.md:222/396; ADR-0001 candidates; next_priorities.md:86; stack_currency_audit
- **Cisco security-LLM orchestrator A/B (Foundation-Sec-8B)** — Swap the grounded orchestrator for an open security LLM on networking/security corpora.  
  *Unblock:* VERIFIED->DEFER (sibling of P2.2): Foundation-Sec is cybersecurity-only, no open Cisco routing model, no AWQ (must self-quantize), and the security-corpus failures are retrieval/rerank-bound not orchestrator-bound. Revisit only with a verified AWQ + evidence the orchestrator is the bottleneck.  
  *Sources:* ROADMAP.md:200; cisco_security_llm_scope_2026_05_29; ADR-0013 revisit
- **P1.6 — chunker-size default verdict (400 vs 500-600 tokens)** — Decide whether chunk_target_tokens=400 is the right default or 500-600 with a longer model context.  
  *Unblock:* Needs corpus depth (P0 extension) to discriminate; the quality side is eval-corpus-gated.  
  *Sources:* ROADMAP.md:228/611
- **P3.1 — Real-mode benchmark nightly CI (GPU runner)** — scripts/benchmark.py --real measures cold-start + first-token + embedding throughput on a GPU runner; workflow template exists.  
  *Unblock:* Resource blocker: needs an allocated GPU runner (cloud or home rig). Don't pick up without confirming user resources.  
  *Sources:* ROADMAP.md:233/405; next_priorities.md:87
- **VLM-assisted hard-table recovery (table-aware escalation trigger)** — Escalate a Docling-flagged table region to the VLM when GFM extraction fails the header-sanity gate (merged-cell/borderless/rotated/scanned tables fall through both the image-area escalation and Table-RAG); the 10-K Director-Comp under-split drives ar-14/15 false-refuse.  
  *Unblock:* Curator-gated: needs a hard-table corpus where a malformed table causes a should-answer FALSE-REFUSE (the 10-K's malformed-table queries are counterfactuals, so 'fixing' them risks a hallucination). Column-split shipped for the SQL-store path; the general recovery + eval corpus did not.  
  *Sources:* ROADMAP.md:400; next_priorities.md:70; verify_numeric_backstop; table_sql_robustness; src/memex CLAUDE.md
- **cross_encoder under-ranking of answer-dense tables (margin-bounded table promotion)** — Force answer-dense GFM tables into top_k when the reranker prefers a more-query-relevant misleading diagram chunk (margin-bounded promotion / top_k bump).  
  *Unblock:* A real deferred lever declined as a risky global retrofit for a cosmetic issue; gated on evidence it helps without regressing prose corpora.  
  *Sources:* ROADMAP.md:398; vlm_state_diagram_limit_2026_05_26
- **Specificity-ranked expand_graph re-introduction at large-corpus scale** — Re-enable graph expansion in the RAG path using related_documents IDF×kind specificity ranking (not unranked neighbors()); the hook is documented but unshipped.  
  *Unblock:* At 47 docs k=50 recall is near-total so expansion adds nothing (default-OFF, A/B byte-identical). Blocked until the corpus grows large enough that hybrid retrieval demonstrably misses relevant docs.  
  *Sources:* ADR-0011; graph-discovery.md; db_audit_2026_05_28; ROADMAP.md:372
- **✅ Citation-chain following (transitive multi-hop CITES traversal) — SHIPPED 2026-06-14.** `citation_paths()` (`index/graph_store.py`); CLI `memex cites --document D --depth N [--cited-by]` + MCP `citation_paths`. The pre-registered data bar CLEARED: ingesting a 6-paper embedder-lineage citation cluster into the main vault took CITES **6 → 34** (15 academic / ≥5 docs / real multi-hop, BGE→Contriever→SimCSE). Returns each reachable doc at its SHORTEST hop-distance + an example chain in citation order; read-only ⇒ HARD-gate-neutral. Closes the LAST ADR-0011 discovery build-out item.  
  *Sources:* graph-discovery.md § "Transitive chain-following"; ADR-0011 (Update 2026-06-14); graph_store.py::citation_paths; scripts/citation_graph_audit.py
- **Confidence-weighted discovery ranking** — Weight related_documents/co-occurring ranking by OTTER MENTIONS extraction confidence.  
  *Unblock:* MEASURED but NOT shipped: it reshuffles ranking but measures extraction-TYPICALITY not topical SPECIFICITY, so it risks fighting the validated IDF×kind ranking. Unvalidatable without a labelled should-relate gold set; future lever IF a discovery-quality eval is built.  
  *Sources:* graph-discovery.md; ner_leverage_buildout_2026_05_29:22; ROADMAP.md:16; src/memex CLAUDE.md
- **Promote eval-expert judged faithfulness dimensions to gating** — Make the LLM-judged faithfulness dimensions (evidence_fidelity/provenance_honesty) HARD-gating instead of reported-only; add cross-attribution/synthesis eval cases.  
  *Unblock:* Same-4B judge is circular (it missed a false-premise misrepresentation live). Needs a validated non-circular cross-model judge (8B via --judge-model or the reserved MCP flagship) + its own governance; cross-attribution cases need 2 retrievable swappable-fact docs that don't exist yet.  
  *Sources:* ROADMAP.md:202; ADR-0013; expert-eval.md; eval_expert_2026_06_01:27
- **Separable-trace reasoning model for expert mode (enable_thinking split)** — A future model / vLLM --reasoning-parser whose CoT trace can be cleanly split, to expose reasoning in the expert surface; dual-decode kwarg + split_think stay plumbed.  
  *Unblock:* The 4B emits a verbose untagged scratchpad that eats the budget and can't be split, so enable_thinking defaults FALSE. Blocked on a separable-trace model landing.  
  *Sources:* ROADMAP.md:202; ADR-0013 realized-v1; expert_mode_surface_b_2026_06_01
- **Foundation-Sec-8B-Reasoning as the expert-mode security model (REASONER swap-in)** — Wire a domain-specialised security reasoning model into the ungrounded expert/bridge surface via the reserved MEMEX_MODELS__REASONER seam.  
  *Unblock:* No verified AWQ for the security candidate; gated on a 12GB self-quantize prerequisite. The seam is reserved but unused in v1 (uses the resident 4B).  
  *Sources:* ADR-0013 drivers/revisit; ROADMAP.md:202/381
- **Apple Silicon / MLX-LM as first-class deployment target** — Support Apple Silicon (MLX-LM) as a first-class inference target.  
  *Unblock:* Eliminated by the RTX-4070 reference hardware; would require its own ADR. Blocked until Apple Silicon is committed as a target.  
  *Sources:* ADR-0001 alternatives + revisit-when
- **Flash-Attention 3 on an FA3-capable reference rig** — Enable FA3 (forbidden on Ada sm_89 due to shared-memory budget) once a Hopper/Blackwell card is first-class.  
  *Unblock:* FA3 needs shared memory Ada doesn't have. Blocked until an FA3-capable reference rig (Hopper/Blackwell) is targeted.  
  *Sources:* ADR-0006 §3 + revisit-when
- **Transformers-loadable / official Qwen3-VL AWQ build (drop the parse-time vLLM dance)** — Run the VLM in-process again if an official Qwen3-VL AWQ or transformers-loadable int4 build appears, dropping the short-lived parse-time vLLM serve.  
  *Unblock:* Today's only int4 build is compressed-tensors (transformers can't run it on 12GB -> decompress-to-dense OOM). Blocked until a transformers-loadable build exists upstream.  
  *Sources:* ADR-0006 §4 amendment + revisit; vlm-vllm-serving.md open-follow-ups
- **Stronger summarizer swap-in at summarize-time (>8B on a bigger GPU)** — Serve a stronger summarizer (Qwen3-14B-AWQ / Gemma-3-12B) briefly at summarize-time via the built-and-gated inference_override swap-in infra.  
  *Unblock:* Infra built + gated OFF; both Gemma-3-12B and Qwen3-14B OOM on the 12GB card even with the orchestrator paused. Blocked on a bigger GPU (or a fitting model); re-enable via MEMEX_MODELS__SUMMARIZER.  
  *Sources:* ADR-0010 swap-in; summarizer_swap_in_2026_05_28; summarizer_model_research
- **Tabular-route figure-framing robustness on complex/mis-bounded tables** — Fix mis-attributed (grounded-but-wrong-metric/period) figures on complex tables in the tabular summarizer + answer routes via better table parsing + label attribution.  
  *Unblock:* Bounded by table-PARSE quality (Docling 10-K bounding), a separate concern — not a hallucination. The next table deepening, not prompt-tuning.  
  *Sources:* ADR-0008 §7; document-summarization.md; ROADMAP.md:162; doc_summarizer_2026_05_27
- **Per-paragraph re-ground pass in report mode** — Run the verifier over synthesized report paragraphs to catch grounding drift in long multi-paragraph bodies.  
  *Unblock:* Deferred until the must_not_assert eval shows drift OR report_confidence trends low (the confidence score is the trigger signal).  
  *Sources:* ADR-0010 §Negative/Revisit/Refinements(4)
- **Residual semantic-overlap cross-paragraph repetition (report dedup)** — Catch purely-semantic re-worded repetition that the deterministic lexical dedup gate misses.  
  *Unblock:* Catching it needs embeddings (non-deterministic under VRAM pressure) or feasible topical headings (absent on decks). Accepted residual; no clean deterministic fix today.  
  *Sources:* ADR-0010 §dedup-gate; deck_granularity_tracker:89-92
- **MAP-loop extraction beyond the 8-claim bridge cap** — Accumulate bridge-grounded claims across windowed passes instead of one 8-claim schema (a wider single schema re-opens the xgrammar force-close trap).  
  *Unblock:* Revisit when a rich analysis regularly saturates the 8-claim cap; the correct fix is a MAP loop, not a wider schema.  
  *Sources:* ADR-0016 §Negative/Alternatives/Revisit
- **Locale/unit coercion ambiguity in table numbers (text-to-SQL)** — Resolve European-decimal-vs-thousands ('1.000') and unit ('5m' metres-vs-million) coercion ambiguity in coerce_number / the WHERE oracle.  
  *Unblock:* Needs context the system lacks; the coercion-soundness guard conservatively refuses rather than guess. Absent from US-format corpora — blocked on contextual signal.  
  *Sources:* ADR-0014 §Negative; table-sql.md robustness-residual; next_priorities backlog-0; ROADMAP.md:254
- **ar-15 borderline Total($)->total-compensation inference** — Make the orchestrator reliably map a bare 'Total ($)' column to 'total compensation' for the MIN-superlative query.  
  *Unblock:* Flips ~1-in-4; the answer node is deliberately NOT over-promoted (that would risk the gate). Model-capability bound.  
  *Sources:* ADR-0014 §Negative
- **Retire the parse pause/serve/teardown dance (orchestrator+VLM one process)** — Eliminate the parse-time vLLM pause/serve by running the VLM co-resident with the orchestrator.  
  *Unblock:* The index/embed phase OOMs the embedder co-resident with any vLLM, so parse still pauses. A separate VRAM-measurement-gated investigation.  
  *Sources:* ADR-0015 §Decision/Negative/Revisit
- **Companion augment node default-ON flip** — Turn the companion retrieval-augmentation node on by default once an eval shows a measured win (ships default-OFF, HARD-gate-adjacent).  
  *Unblock:* Needs refusal_cf=1.0 held over a real transcript<->deck pair + a measured win before default-ON.  
  *Sources:* ADR-0018 §Decision(1); companion_merge_2026_06_04
- **Companion τ_null calibration + (query,doc)/(doc,doc) alignment A/B + DP default-on/λ_jump tuning** — Calibrate the alignment null-floor (companion_align_min_score, default 0.40), run the alignment A/B, and enable+tune the shipped-but-default-OFF MaViLS asymmetric-jump DP / start_s prior. The DP code shipped opt-in; its corpus win is UNMEASURED.  
  *Unblock:* Honestly un-calibrated/unmeasured without a transcript->slide GOLD set (existing 18-frame gold is keyframe->slide). Data-gated on the user labelling which slide is shown when.  
  *Sources:* ADR-0018 §Decision(3)/Amendments; companion-merge.md §3/§13; companion_levers_2026_06_05; companion_merge_2026_06_04; config.py:725-734; companion.py:75/201-205
- **Companion AUGMENT default-ON decision — MEASURED 2026-06-14 → DEFER (do-not-re-walk).** A 15-query joint-grounding eval (`tests/eval-data/companion-augment/`, candidates discovered by a per-pair workflow + content-verified over the 6 linked CR350 transcript↔deck pairs) ran `companion_augment_enabled` OFF vs ON (N=2): **NET-ZERO, double-edged** — answered 8/15 both; refusal_cf=1.0 both; the mechanism fired (13 added events/run ON). One genuine joint-grounding WIN (-03 MAC/OUI refuse→correct) is CANCELLED by a CROWDING regression (-09 rate-limiting correct→refuse: the appended counterparts over-refused a query whose answer was still present). The CR350 modalities are too REDUNDANT (the elaborating modality is retrieved directly) + additive-to-reranked appends dilute borderline queries (the expand_graph / usage_intent_demotion precedent). **`companion_augment_enabled` stays DEFAULT-OFF opt-in** (HARD-gate-safe; infra kept). The DP default-on (`companion_align_dp_enabled`) is SUBSUMED (its sole consumer is default-off). **The crowding-fix was TESTED 2026-06-14 → NO code fix exists:** a COMPETE variant (re-rank window+counterparts, keep window SIZE) A/B'd IDENTICAL to OFF on every query — a companion counterpart can't out-rank the already-best top-k → always cut → augment no-op; the -03 win existed ONLY via append's UNCONDITIONAL entry, so the win + the crowding regression are INSEPARABLE (code reverted). **Revisit ONLY with a genuinely less-redundant transcript/deck corpus** (curator/data, not code). ADR-0018 Amendment 2026-06-14; baselines `_baseline_2026_06_14` + the COMPETE three-way.
- **Companion alignment gold-set authoring (gates τ_null + DP alignment-accuracy A/B)** — Author the transcript→slide gold alignment set. NB it NO LONGER gates the augment default-ON flip (that was MEASURED 2026-06-14 → DEFER above, on the answer-grounding axis, which needs no alignment gold); the remaining use is calibrating τ_null + measuring the DP's ALIGNMENT-accuracy win.  
  *Unblock:* Curator-gated: needs the user to label which slide is shown when across a real lecture pair; none exists yet.  
  *Sources:* ROADMAP.md:5; companion-merge.md §12
- **Re-check companion keyframe floor (0.80) + citation-grade page-map beyond one deck** — Validate/recalibrate the keyframe_min_score=0.80 floor and the citation-grade page-map on more than the single deck (Cours 03) they were tuned on.  
  *Unblock:* Calibrated on ONE deck; re-check on more labelled decks before treating 0.80 as universal.  
  *Sources:* ADR-0018 §Amendment 2026-06-04; companion-merge.md §14
- **Multi-user / multi-GPU webui ingestion** — Scale the exclusive-GPU lock + active-orchestrator reconcile + single-flight beyond one GPU on localhost.  
  *Unblock:* The lock/single-flight/reconcile all assume one GPU on localhost; a 2nd concurrent GPU consumer breaks the model. Blocked until multi-GPU/multi-user is a target.  
  *Sources:* ADR-0019 §Negative/Revisit
- **HEIC/AVIF image-file ingestion** — Accept HEIC/AVIF images in addition to the shipped PNG/JPEG/WebP/BMP/TIFF/GIF (ftyp brands currently excluded).  
  *Unblock:* Needs a separate decode dependency (pillow-heif); deferred until requested often enough.  
  *Sources:* ADR-0020 §Decision(1)/Negative/Revisit; ROADMAP.md:186/70; image-ingestion.md; validation.py:26-28; image_ingestion_shipped_2026_06_05:14
- **Cross-lingual EN-question / FR-corpus rigorous eval** — An EN-question->FR-chunk GOLD set + an EN/FR claim-grounding fidelity matrix to rigorously measure cross-lingual retrieval + grounding.  
  *Unblock:* Live audit shows the behaviour is CLEAN (EN==FR symmetric, no regression), so this is measurement-deepening not a bug fix; the gold set + matrix are not yet built. Flagged a full eval session on its own.  
  *Sources:* next_priorities.md:64; ui_audit_xling_batch_leniency_2026_06_03; audits/11:109-132
- **EN single-token chart-reference artifact scope** — Resolve a single-token EN chart reference (e.g. 'TSMC chart') to its doc in the artifact-scope resolver (currently takes the full-corpus path via the single-token gate).  
  *Unblock:* Stays deferred pending a structural 'doc has a chart/diagram near the qualifier' signal — the FTS chart-strip blind spot makes the in-chart token invisible.  
  *Sources:* artifact-scope.md §Anti-scope; artifact_scope_256:27
- **W10 two-column reading-order reorder** — Custom (column,y) bbox reorder for two-column papers whose reading order scrambles.  
  *Unblock:* DATA-GATED: pymupdf4llm already reads the vault's two-column papers correctly (cosmetic §5 scramble, 0 answer-eval impact); high blast-radius. Unblocks on a curated two-column doc whose scramble causes a measurable answer regression.  
  *Sources:* ROADMAP.md:382; audits/10:178; next_priorities.md; build_status.md
- **W9 born-digital equation handling / equation_count manifest truth** — OCR-LaTeX transcription of born-digital equations + an honest manifest equation_count (currently hardcoded 0).  
  *Unblock:* OCR-LaTeX is a heavy separate model; equation refs near-absent on these docs (gte 1, NIST 0). Deferred until a chart/equation-dense corpus or an OCR-LaTeX pipeline lands.  
  *Sources:* ROADMAP.md:381; audits/10:180; chart_sidecar_2026_06_02:18; pymupdf_worker.py:754-757; pipeline.py:2407
- **AcroForm/widget form-field extraction** — Read PDF form-field (AcroForm/widget) semantics (mined from DocuFlow form_detection.rs).  
  *Unblock:* Out-of-scope roadmap; only if a forms eval category is pursued.  
  *Sources:* ROADMAP.md:255 (Tier 6)

## 📋 Feature backlog (64)

_Would-do, lower priority — no external blocker, just unscheduled._

- **Grounding-gate over-refusal — CITATION-class FIXED 2026-06-06, SUBSUMED by the audit-12 advisory-gate rebalance 2026-06-08 (ADR-0022); SYNTHESIS-class remains.** **UPDATE 2026-06-08:** the citation-floor fix below was broadened into the full advisory-gate rebalance (ADR-0022, `0d76ace`, `docs/audits/12`): `assess_sufficiency`→**v4** (light pre-filter, supersedes the v2 citation-floor), `answer`→**v5** (subject-presence), `assess_relevance`→**v2** (world-knowledge ban) — **false refusals 30→18, +12 ANS, refusal_cf=1.0 N=3**. The SYNTHESIS-class (below) is the documented residual that the rebalance also defers. The original citation-class record: the assess (sufficiency) gate over-refused with the answer present (handwritten-04: answer at rank #1, refused "lack specific citations as requested"). **FIXED** via `assess_sufficiency@v2` (citation-floor prompt; branch `fix/assess-sufficiency-citation-floor`): multi-run validated +2 ANS / 0 regressions / refusal_cf=1.0 across two full 12-corpus passes. The candidate that ALSO tightened ("must explicitly state / topic overlap") was net −3 and rejected. **Residual = the SYNTHESIS-class** (cr350-img-01). **INVESTIGATED 2026-06-06 → NO SAFE ADDRESSABLE SPACE, no code written** (see the closed item under 🚫 Decided-against + `synthesis-lever-nogo-2026-06-06`). The strict gate ALREADY grounds pure-(A) co-located joins; the only refusals are reading-(B) premise-joins / false inferences (it correctly killed a 10-K "50x lower cost/token" claim when the chunk said 35x — the hole). Don't reopen as a gate-relaxation.  
  *Sources:* synthesis-lever-nogo-2026-06-06; grounding-gate-overrefusal-2026-06-06; agents/answering.py::assess; prompts/assess_sufficiency/v2.md
- **De-hyphenation markdown cleanup** — ✅ MEASURED DEFER 2026-06-07 (the "verify first" probe is DONE). Low-risk post-process to rejoin end-of-line-hyphenated words to improve embedding + BM25 token matching. **The probe the entry demanded returned 0 occurrences across all 62 vault `.md`** (strict `word-`+newline+`word`, incl. accented + the highest-risk born-digital justified-prose docs NIST-800-207 / the 10-K): the artifact is **structurally absent** — pymupdf4llm + Docling already merge physical PDF lines into one logical line per paragraph, so there is no `word-\nword` residue to rejoin. The only end-of-line hyphen present is a URL fragment a rejoin would wrongly mangle; the 38 inline `x- y` hits are legitimate suspended compounds / Cisco terms / FR hyphenates. No beneficiary → no code; an `.md`-body rewrite would also churn content-addressed chunk_ids (re-index + a REAL eval re-baseline). **Re-runnable revival** only if a future ingested doc actually exhibits `word-\nword` residue.  
  *Unblock (re-runnable):* re-grep the vault for `word-`+newline+`word`; build only if a doc exhibits it. Tier 6.  
  *Sources:* ROADMAP.md:248; audit wf_76f65abf (de-hyphenation verdict)
- **Adaptive batch-size autotune (rerank/VLM OOM-backoff)** — ✅ DONE 2026-06-07 (rerank). `retrieve/rerank.py::_score_with_oom_fallback` now does a bounded GEOMETRIC backoff (halve on CUDA-OOM → `_empty_cuda_cache` → retry, floor `_MIN_RERANK_BATCH=1`) so it lands on the LARGEST batch that fits (8→4 ≈ 2× the old one-shot 8→1) instead of collapsing to 1; re-raises at the floor or on a non-OOM error. Correctness-neutral (batch size is compute-grouping only; both backends re-score all pairs each attempt) ⇒ no eval needed. Pinned by `tests/unit/test_rerank.py` (geometric-to-floor / lands-on-largest-fitting / non-OOM-reraise). The VLM "half" is N/A (vLLM-served now — no in-process batching); embedder OOM-backoff is a separate, unrequested path (out of scope).  
  *Sources:* retrieve/rerank.py; tests/unit/test_rerank.py
  *Sources:* ROADMAP.md:250/401
- **Coordinate/whitespace-gap borderless-table detection + table-quality confidence gate** — Geometry fallback to recover borderless tables Docling emits as ragged text + a confidence score to gate SQL-vs-linearization (mined from Intellidoc/DocuFlow).  
  *Unblock:* Tier 6; addresses the residual borderless-table answerability gap at the geometry level. The genuinely-borderless-REAL-doc table case is the headline Tier 6 item.  
  *Sources:* ROADMAP.md:251
- **Multi-signal heading-level scoring with caption-penalty** — Richer heading formula: font-weight + length + vertical-isolation + numbering/case + explicit Table N/Figure N caption-penalty (mined from Intellidoc).  
  *Unblock:* Tier 6; would harden Docling level-recovery against caption-mis-promotion / prose-mis-detection.  
  *Sources:* ROADMAP.md:252
- **Table caption/footnote association to [table-rows]** — Attach a table's caption (+footnotes) to its linearized [table-rows] block for higher-signal retrieval (mined from Intellidoc).  
  *Unblock:* Tier 6; a separate retrieval-signal lever from the shipped bold nearest_table_caption (which is SQL-disambiguation only).  
  *Sources:* ROADMAP.md:253
- **European-locale numeric format + context-scoped OCR-digit repair** — Extend coerce_number for 1.234,56 and digit-confusion repair (l->1, O->0) only in numeric context (mined from Intellidoc).  
  *Unblock:* Tier 6, niche (European financials / chart-OCR); also the documented residual of the text-to-SQL robustness work.  
  *Sources:* ROADMAP.md:254/159; table_sql.py:440
- **Richer Intellidoc parse-eval metrics + severity-ranked error taxonomy** — Table-boundary IoU, merged-cell P/R, markdown tree-similarity + a severity-ranked error taxonomy beyond the shipped structural-F1 facets.  
  *Unblock:* Optional refinement; the core cell-content P/R facets already closed the open sub-goal.  
  *Sources:* ROADMAP.md:247 (Tier 6)
- **Roadmap Tier 6 predecessor-mined parser bundle (rollup)** — The DocuFlo/DocuFlow/Intellidoc-mined structural-heuristic backlog as one tracked rollup: autotune, borderless-table detection, heading scoring, caption association, locale/OCR numeric repair, AcroForm. (Individual items also listed.)  
  *Unblock:* Genuine Tier 6 roadmap candidates, not yet built; pick up per-item when the relevant eval category is pursued.  
  *Sources:* p0_corpus_nist_2026_05_25:33; build_status.md (Tier 6)
- **W9 VLM/OCR transcription of dropped born-digital figures** — Escalate silently-dropped born-digital figures to VLM/OCR transcription rather than only a bare <!-- image --> visibility placeholder.  
  *Unblock:* The heavier escalation beyond the shipped visibility placeholder; deferred as heavier work.  
  *Sources:* audits/10:180
- **Docling running-header furniture strip (multi-line-aware band)** — Extend the repeating page-furniture strip to docling-deck running headers with a multi-line-aware band.  
  *Unblock:* Decks rarely have running headers; deferred until a deck case appears.  
  *Sources:* audits/10:158
- **Rich document view (original PDF / clean raw md / rich render)** — Side-by-side or toggled rich document view that the audit-10 parse-stage cleanup was the precursor for.  
  *Unblock:* The cleanup precursor shipped; the rich-view UI itself is the unbuilt payoff.  
  *Sources:* audits/10:7/67-73; ROADMAP.md:380
- **download-models.py implementation** — ✅ DONE 2026-06-07. The stub is now a real model-bootstrap CLI: `resolve_model_targets` reads the configured ids from `MemexSettings` (core: orchestrator/embedder/reranker; gated: VLM/chart-OCR/summarizer/OTTER/ASR — `--all` for the full kit; `reasoner` skipped), fetches each into the HF cache via `huggingface_hub.snapshot_download` (faster-whisper's `download_model` for the ASR CT2 repo — same cache; OTTER's transitive `config.token_encoder` repo fetched too), reports per-model size + total, exits 0/1/2 (all-ok / any-missing-or-failed / setup-error). `--check` verifies the cache offline (`local_files_only`), `--json`/`--only` for scripting. The one online bootstrap step for the air-gap workflow. Live-validated (`--check` 6/6 present on the live config; `--check --all` includes VLM); pinned by `tests/unit/test_download_models.py` (12 tests, faked hub/faster-whisper — no network). **E2E-WIRED 2026-06-07:** the logic was promoted to the package module `src/memex/models/download.py` (`run_download`/`format_report`/`model_cache_status`; re-exported from `models/__init__`); the script is now a thin shim over it; added the discoverable `memex download-models` CLI command (CUDA-free, skips `bootstrap()`) + a read-only webui `/resources` **Model cache** status panel. Tests retargeted to the module (20 tests) + a CLI-command test + webui panel tests; live Chrome e2e of /resources (6/6-cached + missing states).  
  *Sources:* src/memex/models/download.py; scripts/download-models.py; src/memex/cli/commands.py (`download-models`); src/memex/webui/app.py (`_models_panel`); tests/unit/test_download_models.py; tests/integration/test_download_models_cli.py
- **tiktoken-counted chunk tokens** — Swap the chunker's word-count 'tokens' for real tiktoken-counted tokens (current word-count is ~1.3x lower than real transformer tokens).  
  *Unblock:* On the roadmap as a future swap; couple to the P1.6 chunker-size verdict.  
  *Sources:* core/config.py:558-560
- **SQLite/LanceDB connection reuse (long-lived handles)** — ✅ MEASURED DEFER 2026-06-07 (`scripts/db_open_bench.py`, the docling-NO-GO pattern). The premise ("FTS 3-5x + LanceDB 2-4x per /ask is the highest-leverage perf item") is FALSIFIED by measurement: on the warm 124-doc vault the per-store opens are **LanceDB 0.53ms / FTS 0.22ms / Table 0.15ms** — opens are **3.9% of a real `hybrid_search`** (open 2.0ms vs embed 25.7 + search 22.6) and ~0% of the LLM-dominated /ask. Reusing them buys ~2ms for the silent stale-inode hazard (a held sqlite handle reads the old unlinked inode after `reindex --force`). NO reuse code shipped. **The ONLY non-trivial open is `GraphStore.open` = 21ms** (the embedded ryugraph DB cold-open graph_store.py:48-49 explicitly pre-deferred), which is ~47% of `related_documents` (44.7ms) / ~20% of `entity_overview` (106ms) — but it's STILL a defer: (1) single-user local-first app → no throughput dim, 21ms is imperceptible inside the webui HTTP+render+paint; (2) NO hot-loop consumer (`expand_graph` default-OFF; every other site is once-per-interactive-request, and CLI one-shots open once then exit → zero reuse benefit; `related_documents_for_seeds` already hoists the open out of its seed loop); (3) the safety story is UNVERIFIED — ryugraph is single-writer embedded with no known `read_consistency_interval` analogue, so a reused webui read-conn could serve STALE graph data after an enrich/reindex (today there's no staleness precisely because every request reopens).  
  *GraphStore reuse — now CONCLUSIVELY PRECLUDED (ryugraph read-consistency root-caused 2026-06-07, `scripts/ryugraph_consistency_probe.py`):* the "unverified staleness" concern was MIS-FRAMED — it's an EXCLUSIVE LOCK, not staleness. ryugraph 25.9.2 takes an exclusive lock on the DB directory on ANY open (read_only OR read-write); cross-process, a held handle locks out ALL other-process opens (`RuntimeError: IO exception: Could not set lock on file`). So a process-lifetime cached webui handle would block EVERY concurrent `enrich`/`index`/`reindex`/`retitle` — architecturally impossible, no `read_only` escape. The current **open-per-request** design is CORRECT + necessary (probe R3: a fresh reopen sees cross-process writes; in-process concurrent opens coexist — R6 — so the lock is per-process). The 21ms `Database()` build is the irreducible price of the lock model. The "revisit trigger" above is VOIDED for GraphStore — reuse is off the table regardless of a future hot-loop consumer.  
  *DISCOVERED (separate, pre-existing) robustness gap — ✅ FIXED 2026-06-07 (both sides):* the lock-contention `RuntimeError` was UNCAUGHT (only `ImportError` was), so a brief cross-process race CRASHED LOUDLY — it would 500 a webui discovery read (REACHABLE: the discovery GET routes doc-view/`/entity`/`/graph` are NOT behind `_ingest_guard`, so a UI-ingestion enrich subprocess could 500 a browse) or fail an `enrich`/`index` graph step. Fixed via two policy helpers in `index/graph_store.py` — `open_graph_for_read` (FAIL-OPEN → None, so a discovery read degrades gracefully) + `open_graph_for_write` (bounded-RETRY to ride out transient contention, then RE-RAISE if persistently locked — a writer fails LOUD, never silently skipping) — gating on the narrow `is_graph_lock_error` (MESSAGE match, NOT bare `RuntimeError`, which also wraps real corruption/schema → those still propagate). Wired into the retrieve helpers (`entity`/`related`), MCP reads, the webui doc-view + `/graph` (except-widened in place to keep the `GraphStore` test seam), `agents/answering.py::expand_graph` (read, fail-open→skip; default-OFF, reviewer-caught), and the writers (`enrich`/`_open_graph`/`link-slides`). The 3 CLI one-shot discovery commands (`graph`/`related`/`cites`) deliberately keep RAISING (an explicit "graph busy" beats a misleading empty result for a user-paced one-shot). Pinned by `tests/integration/test_graph_lock_resilience.py` (real cross-process holder) + webui lock-fail-open/propagate + expand_graph fail-open tests. See memory `ryugraph_consistency_probe_2026_06_07`.  
  *Sources:* db_audit_2026_05_28:32/17; graph_store.py:48-49; scripts/db_open_bench.py; scripts/ryugraph_consistency_probe.py; memory `db_connection_reuse_nogo_2026_06_07` + `ryugraph_consistency_probe_2026_06_07`; ROADMAP.md:372
- **Cross-document table SQL** — Query tables.sqlite corpus-wide (compare revenue across all annual reports) instead of the per-doc in-memory store.  
  *Unblock:* Deferred lever gated by the existing row-verbatim/recompute fabrication boundary; not built. (Anti-scope for table-sql v1 per spec.)  
  *Sources:* db_audit_2026_05_28:37; table-sql.md §Anti-scope
- **Table-SQL group-by / cross-table joins / derived-superlative aggregates** — Support GROUP BY, cross-table JOINs, and derived-superlative aggregates in the text-to-SQL path (currently -> refuse/no-op).  
  *Unblock:* Revisit only if the independent recompute oracle can be extended to cover them SAFELY (never delegate the WHERE to sqlite).  
  *Sources:* table-sql.md §Anti-scope/§4
- **Promote table-chunking constants to an IndexSettings field** — Make MAX_CHUNK_CHARS / multiplier tunable IndexSettings fields instead of module constants.  
  *Unblock:* Anti-scope for v1 (module constants); promote later if tuning is needed.  
  *Sources:* table-chunking.md §Anti-scope
- **Docling per-page header-aware export path** — Apply the header-aware table serializer to the per-page export path (not just the whole-doc export).  
  *Unblock:* Left as the fallback path if the per-page object doesn't support it.  
  *Sources:* table-header-reattach.md §Part 2
- **Merged-column split false-positive hardening (coordinates, mean±stddev)** — Harden split_merged_columns against a proven-but-absent false-positive class.  
  *Unblock:* Currently contained (recompute-gated, kill-switchable, 0 false-splits on 47 docs); the class is proven-but-absent in current corpora. Harden when a corpus exhibits it.  
  *Sources:* ADR-0014 §Negative
- **Numeric-aggregate backstop in shared ground_claims (bridge + summarizer)** — Wire the /ask verify-node deterministic numeric-grounding backstop into the shared ground_claims helper.  
  *Unblock:* The demotion lives in the /ask verify NODE not the shared helper; bare computed-table-figure claims are out of v1 bridge scope. Revisit when a numeric-heavy bridge use emerges.  
  *Sources:* ADR-0016 §Negative/Revisit
- **--ground flag on expert mode** — Add a --ground flag to expert mode as a lighter alternative to the dedicated bridge surface.  
  *Unblock:* Rejected-for-v1 in favour of a dedicated surface; a trivial future addition if wanted.  
  *Sources:* ADR-0016 §Alternatives
- **Reasoning-over-retrieved grounded-synthesis (supported-by-evidence-set gate)** — 🚫 **INVESTIGATED 2026-06-06 → NO SAFE ADDRESSABLE SPACE (no code).** Attempted via a safety-first staged plan; NO-GO at the Phase-A authoring checkpoint. A 22-candidate strict-gate probe (6 domains) showed the gate ALREADY grounds pure-(A) co-located joins (20/22 are literal-reads; a by-construction test grounded clean transitive joins 3/3), and the only refusals are reading-(B) premise-joins / FALSE inferences (the decisive one: a 10-K candidate inferring "50x lower cost/token" when the chunk states 35x — relaxing to accept it reproduces the contextual-retrieval hole). The "synthesis" space has no safe middle: co-located joins are already grounded; refused inferences need an unstated premise = the hole. **Do not reopen as a gate-relaxation.** Value isn't zero — the bridge already surfaces these as labelled-ungrounded analysis today.  
  *Sources:* synthesis-lever-nogo-2026-06-06; grounding-gate-overrefusal-2026-06-06; contextual-retrieval-negative-2026-05-25
- **MCP flagship-model fallback layer** — A separate upstream MCP layer that escalates to a more-powerful remote flagship model (the user's stronger models).  
  *Unblock:* Reserved/scoped but not built; inverts local-first/air-gap so needs explicit/labeled/consented escalation + its own governance (likely its own ADR).  
  *Sources:* mcp_scope_directive_2026_06_01; src/memex CLAUDE.md
- **scan doc-type summarization route** — Specialise the scan route (over VLM text) in the document summarizer; currently routes as generic 'long'. short/long/tabular/deck all shipped.  
  *Unblock:* Corpus-gated; slots into _classify_route + a per-route MAP prompt. Revisit when a scan-summary need arises.  
  *Sources:* ADR-0008; document-summarization.md; scan-vlm-parse.md; ROADMAP.md:164; document_summarizer.py:248
- **Executive-summary-over-the-body layer (report mode)** — A second bounded reduce over the batch paragraphs to produce an exec summary.  
  *Unblock:* Forcing-function-gated: awaits a real need for an exec-summary layer above the report body.  
  *Sources:* ADR-0010 §Revisit
- **Theme/salience paragraph clustering in report mode** — Cluster report paragraphs by theme/salience instead of document-order adjacency.  
  *Unblock:* Would need another model call; a possible refinement over the deterministic adjacency batching.  
  *Sources:* ADR-0010 §Neutral
- **Summarizer publication-metadata key-point suppression** — ✅ DONE 2026-06-07 (`agents/document_summarizer.py::_select_doc_key_points`). **A live N=3 probe (the original premise was STALE — attributed to the 8B; re-measured on the 4B) confirmed the residual REPRODUCES deterministically AND is worse than the static read assumed:** the NIST headline was 7/12 publication-metadata (FISMA / ITL / trademark / patent / ToC / Federal-CIO / "Section 2 defines") and contained NONE of the tenets / PE-PA-PEP components / trust algorithm — the v2 MAP prompt's "return zero key_points for front-matter" is not reliably obeyed by the 4B, so the front-matter sections (first in reading order) exhausted the round-robin cap before the body. **Fix = deterministic BODY-FIRST selection** (`_is_front_matter_section` encodes the v2 prompt's metadata enumeration as a frozenset of universal labels — Authority/Acknowledgments/Trademark/Patent/ToC/List-of-*/Keywords/References/…; `_select_doc_key_points` round-robins CONTENT sections first, front-matter only as a FALLBACK so the headline is never empty). Same pattern as the verify numeric/name-only backstops (prompt asks → deterministic filter enforces). Selection-only ⇒ HARD-gate-neutral (per-section breakdown + the abstract untouched; the abstract was already excellent). **Re-probe: NIST headline 7/12-metadata → ~1/12, now leads with the ZT definition + the "all data sources are resources" tenet + the Policy Engine + the 3 implementation approaches.** Validated by the REAL `memex eval-summary` ×2 (6/6 summarize_correct, mean_recall 1.0, 0 leaks — unchanged). Pinned by `tests/unit/test_doc_type.py` (`_is_front_matter_section` matrix + body-first/fallback/never-empty + the existing round-robin tests unchanged).  
  *Sources:* next_priorities.md (Summarizer follow-ups)
- **Per-card-tier VRAM calibration tables (8/16/24 GB)** — Per-tier resolver tables so the mode system calibrates to 8/12/16/24 GB cards (curated constants are 12GB-calibrated; other tiers stay conservative).  
  *Unblock:* Revisit when a second card tier is calibrated.  
  *Sources:* ADR-0007 §Expanding/Negative/Revisit
- **New capability modes (throughput/batch, scan_ocr, index_only) + more ResourceProfile knobs** — Curated co-residence modes (throughput/batch sweeps, scan_ocr co-resident VLM, index_only no-orchestrator) + additive ResourceProfile fields (rerank batch size, KV dtype, quant tier, independent embedder device).  
  *Unblock:* Named candidates for the horizontal/vertical mode axes; additive (unset defaults to today's behaviour). Not yet built.  
  *Sources:* ADR-0007 §Expanding(Horizontal/Vertical)
- **SOURCE parse-artifact entity junk (NEMOCLAW-class)** — Residual junk entities that are in-document parse artifacts, separately tracked as a parse-quality issue.  
  *Unblock:* Not OTTER mis-extraction; a parse-quality concern. Fix at the parse stage.  
  *Sources:* ADR-0012 §Neutral
- **DEFINES/RELATES_TO relation edges + OTTER∪LLM fusion** — Populate DEFINES/RELATES_TO relation edges and fuse OTTER NER with the LLM extractor.  
  *Unblock:* Needs a relation-extraction stage OTTER isn't; OTTER-alone already wins the A/B; graph-only payoff at this scale. Revisit at large-corpus scale.  
  *Sources:* graph-discovery.md; ADR-0012; audits/08; src/memex CLAUDE.md
- **Multi-LoRA serving for ad-hoc orchestrator experimentation** — Use vLLM multi-LoRA serving to swap orchestrator variants without restarting the server.  
  *Unblock:* Partially mitigates the one-model-per-process constraint but adds complexity the team chose to defer.  
  *Sources:* ADR-0001 §Consequences
- **Extract inference layer as a standalone package** — Extract a module (likely the vLLM inference helpers) into a separately-publishable Python package.  
  *Unblock:* Single-package decision; revisit when the team grows past 3-4 contributors or a module has external demand.  
  *Sources:* ADR-0002 §Consequences/Revisit
- **Binary-export feature for users who bypass Markdown** — Add a binary-export feature if a meaningful contingent of users wants to bypass Markdown.  
  *Unblock:* Forcing-function-gated; would not change the architecture.  
  *Sources:* ADR-0003 §Revisit
- **Cross-machine sync layer over the Markdown vault** — Add cross-machine sync as a layer over Markdown (not a replacement).  
  *Unblock:* Deferred until cross-machine sync is needed.  
  *Sources:* ADR-0003 §Revisit
- **Deterministic Langfuse trace IDs from correlation_id** — Use Langfuse.create_trace_id(seed=correlation_id) so a log-line ID is directly findable in the UI.  
  *Unblock:* An optimisation, not correctness; adopt when there's a running Langfuse server to validate against.  
  *Sources:* ADR-0004 §Operational-Notes
- **P4.3 — Trace retention windows (Langfuse self-host)** — Match the EventBus 30-day prune retention to Langfuse self-host retention windows once self-host lands.  
  *Unblock:* Langfuse self-host wiring is open (langfuse_enabled False by default); open architectural Q5.7. Match when self-host lands.  
  *Sources:* ROADMAP.md:239/594/405; next_priorities.md:88
- **--language-model-only / vLLM 0.22 bump for the text-only orchestrator role** — Shrink the 4B orchestrator's footprint by loading text-only (drops the vision tower) via vLLM 0.22's --language-model-only.  
  *Unblock:* Not load-bearing today (the 4B fits on 0.21.0). The flag is unwired with a fragility history (#36275) and needs a fresh guided-JSON conformance probe on the changed topology. Couple the 0.22 bump to a future swap, MEASURE-FIRST.  
  *Sources:* ROADMAP.md:224; ADR-0015 §Revisit; qwen35_4b_orchestrator_swap; qwen_migration_research
- **P2.3-b hoist one VLM-vLLM across a bulk re-ingest** — Keep a single VLM-vLLM process up across a bulk re-ingest instead of short-lived parse-time spawns (avoids ~30s/doc startup).  
  *Unblock:* Deferred: reintroduces the chart-OCR co-residence pressure (7.4+3 GB > 12 GB). Fresh-process-per-doc is the safe pattern.  
  *Sources:* ROADMAP.md:398; vlm-vllm-serving.md; qwen3vl_migration_resume:18
- **P3.3-a Qwen2.5-VL chart-OCR retry** — Retry the AWQ Qwen2.5-VL as a chart-OCR backend now that the PytorchGELUTanh rename is handled.  
  *Unblock:* UNBLOCKED 2026-05-25 (AWQ load compat shipped), pickable but not picked. Durable fix still wanted: retire deprecated AutoAWQ / the Qwen3-VL upgrade.  
  *Sources:* ROADMAP.md:399
- **P3.3 v7 follow-up (a) — chart-types-09 multi-row Gantt-assignee table false-refuse** — A multi-row Gantt-assignee table query still false-refuses under chart-OCR.  
  *Unblock:* Open (not blocking). Possible angle: per-cell-pair prompt rendering or cross-product table reformat.  
  *Sources:* ROADMAP.md:393
- **P3.3 v7 follow-up (b) — Q05 prose false-refuse under chart-OCR+v7** — A prose query (bar charts + maps) false-refuses under chart-OCR+v7 but answers under PyMuPDF.  
  *Unblock:* Open (not blocking), n=1; likely chart-block-in-verifier-view perturbation; needs more reps to confirm.  
  *Sources:* ROADMAP.md:394
- **VLM best-of-N completeness convergence (N>=2 default)** — Raise vlm_transcription_samples default from 1 so the longest (most-complete) draw is cached, not the first.  
  *Unblock:* Default N=1 caches the FIRST draw (not necessarily most-complete). Shipped as opt-in; the default was not raised.  
  *Sources:* vlm_path_revival_2026_05_25:35
- **VLM cite-precision anchor re-authoring** — Re-author per-query eval anchors against the VLM-re-chunked vault (cite-prec dropped cr350 0.933->0.702; xref-03/04/09 at 0.0).  
  *Unblock:* Deferred follow-up if cite-prec matters: the drop is a re-chunking artifact (answers correct, agent grounds in valid sibling/new chunks); would need per-query anchor re-authoring against the new chunking.  
  *Sources:* vlm_path_revival_2026_05_25:44
- **Lecture-summary path + speaker diarization (ASR follow-ons)** — A dedicated lecture-summary path over transcribed audio + pyannote who-spoke diarization.  
  *Unblock:* Natural follow-ons noted as the audio-ASR route shipped; low value for single-instructor lectures (we ground on text). pyannote is HF-gated -> must provision before air-gapping.  
  *Sources:* ROADMAP.md:148/7; ADR-0017 §Negative/Revisit; audio-asr-route.md §15; asr_backend.py; asr_audio_scope_2026_06_03:28/50
- **tests/eval-data/audio-*/ WER corpus** — A word-error-rate eval corpus for the ASR ingestion route.  
  *Unblock:* Listed under ASR deferred follow-ups; turbo meets the bar so it's low-urgency, but no WER corpus has been authored (needs curated audio + transcripts).  
  *Sources:* audio_video_asr_shipped_2026_06_03:30; asr_audio_scope_2026_06_03; next_priorities.md:30
- **Synced audio player UI + LLM-titled transcript pass** — A webui audio player synced to the [mm:ss] anchors, and an LLM-generated transcript title (vs filename-derived frontmatter title).  
  *Unblock:* Nice-to-have, not v1; the deterministic retitle writer exists but no title generator.  
  *Sources:* audio-asr-route.md §7/§12/§15
- **Per-kind ingest size policy (lift cap only for media)** — Replace the single global ingest.max_bytes 2 GiB ceiling with a per-kind size policy.  
  *Unblock:* For now one global ceiling; a future per-kind policy could lift only media.  
  *Sources:* ADR-0017 §Amendment
- **whisper.cpp (ggml) low-resource ASR fallback** — Add whisper.cpp as a lightest-runtime low-resource ASR fallback.  
  *Unblock:* Another distinct runtime + weaker word timestamps; a fallback, not the default.  
  *Sources:* ADR-0017 §Considered-Options(7)
- **Companion-merge auto-pairing (transcript<->deck, course-code/ordinal)** — Auto-pair a transcript to its slide deck by course-code/ordinal inference instead of the explicit link-slides CLI.  
  *Unblock:* Auto-pairing is non-trivial (Cours-04 <-> Semaine-4) and a wrong pair mis-attributes commentary. Deferred to a SUGGEST-only layer via the course_refs resolver precedent.  
  *Sources:* ADR-0018 §Decision(4); companion-merge.md §4/§13; companion_merge_2026_06_04
- **Companion-merge segment-level alignment** — Align at the finer manifest TranscriptSegment granularity instead of the chunk level.  
  *Unblock:* A deferred refinement; chunk-level keeps alignment in the surfaces'/augmentation's unit.  
  *Sources:* companion-merge.md §2/§13
- **Companion-merge stored-embedding reuse** — Reuse LanceDB stored embeddings for alignment instead of re-embedding both sides on demand.  
  *Unblock:* Deferred optimization; LanceDB exposes no raw stored-vector export and the merge is an offline op (cost acceptable).  
  *Sources:* companion-merge.md §2/§13
- **Companion-merge title/agenda-slide downweighting** — Downweight a title/agenda slide that matches everything in the alignment.  
  *Unblock:* Listed in the deferred bundle; not built.  
  *Sources:* companion-merge.md §13
- **Companion-merge multi-frame keyframe sampling + keyframe match into DP transition cost** — Sample a couple frames around the time-range midpoint and keep the best match (transition frames); fold the keyframe match into the monotonic-DP TRANSITION COST (beyond fixing keyframe-PRIMARY chunks as anchors).  
  *Unblock:* Deferred within the keyframe-OCR lever; the DP already fixes keyframe-PRIMARY chunks as anchors.  
  *Sources:* companion-merge.md §14
- **Citation-grade deck page-map migration (per-doc re-parse activation)** — Bring all docs to citation-grade page mapping; an escalated deck must re-parse with VLM enabled (foundation wired + validated on one deck + merged).  
  *Unblock:* Per-doc on next re-parse+reindex (not auto-applied); a documented follow-on like every re-parse migration.  
  *Sources:* ADR-0018 §Amendment 2026-06-05; next_priorities.md:12
- **Non-VLM fast printed-text screenshot OCR route** — Add a fast printed-text-only OCR path for screenshots as an additional image route.  
  *Unblock:* An additional route, not a change to the VLM-mandatory default. Deferred until worthwhile.  
  *Sources:* ADR-0020 §Revisit
- **Route image parsing by validator kind instead of fixed suffix set** — Route by the validator's detected kind==image so off-list-but-magic-valid extensions (.jfif/.jpe) hit the image branch.  
  *Unblock:* Fixed IMAGE_SUFFIXES covers common extensions; off-list valid images fall through to a clean typed parse failure (not a crash). A future option.  
  *Sources:* ADR-0020 §Revisit
- **Full multi-page TIFF / animated GIF image transcription** — Transcribe every frame of a multi-frame image (v1 takes the first frame only).  
  *Unblock:* Converter-only extension (emit an N-page PDF; the scan route already transcribes every page).  
  *Sources:* ADR-0020 §Negative/Revisit; image_convert.py:46
- **Deterministic blank-image / blank-audio 'no content' meta-response filter** — Recognize a VLM/ASR honest 'this image is blank' meta-response and route it to ParseConfidenceTooLow (no thin doc).  
  *Unblock:* Current behaviour is honest + HARD-gate-safe (thin doc refuses all queries); phrase-matching a meta-response is fragile. Deferred as a vault-tidiness nicety.  
  *Sources:* ADR-0020 §Revisit; image_ingestion_shipped_2026_06_05:27; next_priorities.md
- **Per-claim wikilinks** — ✅ DONE 2026-06-07 (`feat/per-claim-wikilinks`, merged `d997508`). Added `FinalResponse.claim_wikilinks: list[str]` ALIGNED 1:1 with `claims` (entry i = `[[doc#section]]` for claims[i]'s cited chunk; `""` unresolvable), derived in `compose` from the grounded cited chunks — the SAME no-hallucination contract as `wikilinks` (NOT on the LLM-emitted `CitedClaim` schema). NOT deduped (preserves the per-claim mapping vs the flat deduped Sources list); `[]` on refusal; HARD-gate-neutral. **MCP/CLI payload parity** — the webui already renders per-claim sources BY TITLE via `chunk_refs` (the deliberate "sources by title, not raw `[[..]]`" design), so no webui element was added. Validated: 1791 tests + 3 compose tests + live CLI /ask (aligned 1:1) + live webui /ask e2e (answered panel renders per-claim sources cleanly, no regression, clean console).  
  *Sources:* src/memex/agents/answering.py (`FinalResponse.claim_wikilinks` + compose derivation); tests/integration/test_answering_with_fakes.py; wikilink-emission.md §Anti-scope
- **Auto-derived / tag-derived scope sets** — Scope sets auto-derived by tag (e.g. 'all SRWE decks') instead of explicit hand-picked selection.  
  *Unblock:* Anti-scope (a set is an explicit hand-picked selection); tag-derived scoping is a separate future idea.  
  *Sources:* scope-sets.md §Anti-scope
- **Multi-upload ingestion QUEUE (webui)** — Queue concurrent browser ingests instead of single-flight rejecting a 2nd ingest with a 409.  
  *Unblock:* v1 non-goal by contract (single-GPU rig runs one ingest at a time): a design change (job model + per-job GPU scheduling), not hardening. Warranted only on multi-user / multi-GPU.  
  *Sources:* ROADMAP.md:6/183-184; ADR-0019 §Negative/Revisit; ui-ingestion.md; ui_ingestion_deferred_done_2026_06_05
- **Answer-graph stale doc-picker scope clearing (UX)** — Clear/surface a leftover doc-picker scope selection that silently scopes the next /ask -> confusing-but-correct refusal. **✅ CLEAR-HALF SHIPPED 2026-06-07** (`feat/scope-picker-clear-selection`, merged `dee949f`): the always-visible "Clear selection" control (`POST /scope-sets/clear`, no JS) re-renders the picker empty — unchecks manual ticks AND an applied set — + a flash; HARD-gate-neutral (narrow-only `scope_doc_ids`); saved sets preserved; live Chrome e2e'd (2 ticks → 0 + flash, clean console); +2 webui tests. The residual = the count-badge half ONLY (M-effort, below). **VERIFIED 2026-06-07 (audit wf_76f65abf): was partially-shipped + a scope subtlety.** The post-answer surfacing ALREADY ships (the `.ans-scope` "Scoped to your selected document(s)" note, #256/2026-05-27). What remains: (1) an explicit **"Clear selection"** control = clean S-effort (a new `POST /scope-sets/clear` re-rendering `_scope_picker.html` with `checked_ids=[]`, mirroring `/scope-sets/delete`; no JS) — it works for BOTH manual ticks and applied saved-sets (a full server re-render). (2) A **pre-emptive selected-count on the collapsed `<summary>`** is NOT a clean S-fix: it is server-rendered from `checked_ids`, so it reflects only `/scope-sets/{apply,save,suggest,delete}` round-trips — it CANNOT see *manual* checkbox ticks (the actual reported gotcha), because `/ask` swaps `#answer` only and never re-renders the picker. Fully surfacing the manual-tick state pre-ask needs an **OOB picker re-render riding on the `/ask` answer render** (the submitted `scope_doc_ids` are available there) = M-effort, not S. HARD-gate-neutral (presentation over narrow-only `scope_doc_ids` → `resolve_artifact_scope`); no chunk_id churn; webui-test + live-e2e validatable.  
  *Unblock:* Clear-button half DONE. Residual = the pre-emptive count-badge ONLY (M-effort, needs an OOB picker re-render on `/ask` to see manual ticks) — low value, gated on a real need (the Clear control + the post-answer `.ans-scope` note already cover the gotcha). Not auto-uncheck (that defeats saved-scope-set reuse).  
  *Sources:* next_priorities.md:74; audit wf_76f65abf (doc-picker verdict); feat/scope-picker-clear-selection (`dee949f`)

## 🔬 Researched + banked (17)

_Investigated + parked with a verdict; waiting on a forcing function or a measured win._

- **P2.6 — Visual / image-based retrieval (ColPali / ColQwen2-style)** — Embed rendered page images directly (vision embedder + late-interaction) instead of transcribe-diagram->text->EmbeddingGemma.  
  *Unblock:* Reasoned high-cost/low-benefit DEFER: grounding is text-based, recall near-saturated at 47 docs, needs a new index engine + co-resident VRAM. Cheap decisive test = embed already-rendered pages with SigLIP, measure recall@50 uplift. Revisit WHEN the corpus is large+visually-dense AND transcription is the proven bottleneck.  
  *Sources:* ROADMAP.md:229
- **Summarizer model upgrade (Gemma-3-12B-it AWQ swap-in at summarize-time)** — Swap in a stronger model at summarize-time via the parse-time-VLM-vLLM-clone seam — the ONE banked lever from the 3-agent research.  
  *Unblock:* Dead ends (don't re-walk): enc-dec summarizers, decoding levers, vanilla LoRA, any <=8B swap (Qwen3-8B already wins Vectara). The lever IF a future workload shows per-call abstraction is the bottleneck: Gemma-3-12B-it AWQ at summarize-time. NB gemma-3-12b-int4-awq OOMs on 12GB; needs a fitting model or bigger GPU.  
  *Sources:* ROADMAP.md:362; summarizer_model_research_2026_05_28; summarizer_swap_in_2026_05_28
- **OSCAR 2-bit / turboquant / nvfp4 KV-cache quantization** — More aggressive KV-cache quant (OSCAR 2-bit, turboquant_3bit/4bit_nc, nvfp4) for a bigger orchestrator window.  
  *Unblock:* Banked-not-chased: OSCAR is H100/server-scale, SGLang-or-unverified-vLLM, untested on Ada, AWQ-int4-composition + HARD-gate risk; turboquant/nvfp4 are vLLM 0.21 options to revisit only if the window genuinely binds. The 'enable fp8 for a 2x window' premise was already falsified (e5m2 is the live default).  
  *Sources:* ROADMAP.md:190; ADR-0006 §2 amendment; kv_cache_quant_research_2026_05_28:37/3
- **DSPy prompt-optimization for the chat query-rewrite step** — Use DSPy BootstrapFewShot to optimize the multi-turn-chat query-rewrite prompt.  
  *Unblock:* No runtime dep / no ADR; DSPy can't enter the runtime (offline compile-then-bake only) + hand-authored few-shot is the ~90% solution. Unblocks only on the 3-part AND: Surface A shipped + measured >10% follow-up gap + hand-authoring tried-and-insufficient. The A-1.5 gap is now measured ~0%, so it stays deferred.  
  *Sources:* ROADMAP.md:201/202; grounded-agentic-chat.md §9.1; grounded_chat_surface_2026_06_01; chat_multiturn_eval_2026_06_04
- **Unigram Rust tokenizer for the reranker** — Perplexity's Rust Unigram tokenizer applied to the XLM-RoBERTa reranker.  
  *Unblock:* Thematic bullseye but LOW-leverage (tokenization ms vs ~20s CPU forward pass); banked, not chased.  
  *Sources:* unigram_tokenizer_research_2026_05_28; ROADMAP context
- **BERT-NER / GLiNER enrich extractor swap (discovery-quality)** — Swap the OTTER/LLM entity extractor for a heavier BERT-NER (GLiNER) / fine-tune / OTTER∪LLM fusion to fix residual entity noise (STP/CR350 mis-typed, junk ports, FR connectors) — the root-cause fix.  
  *Unblock:* OTTER-alone already wins the A/B (fusion currently moot); blast radius doesn't justify a training pipeline. Gate: only swap once discovery QUALITY is proven the bottleneck, and only via a hand-labelled should-relate gold-set A/B. OTTER shipped; the GLiNER A/B did not.  
  *Sources:* ADR-0012 §Alternatives/Revisit; bert_ner_enrich_scope_2026_05_28; entity_centric_retrieval_2026_05_28; next_priorities.md
- **Qwen3-Reranker-0.6B backend measurement (smaller-model headroom)** — Measure the wired-but-unmeasured reranker_backend=qwen3 (0.6B) path; smaller would widen GPU headroom and the dynamic-VRAM seam could auto-pick cuda-vs-cpu.  
  *Unblock:* Banked future lever: the path is wired but the smaller-0.6B variant is unmeasured (distinct from the P2.1 0.6B quality A/B which lost). Reduced urgency by the dynamic-VRAM manager.  
  *Sources:* reranker_gpu_ab_2026_06_01:26; build_status.md
- **Qwen3.5-9B / Qwen3-4B-Instruct-2507 orchestrator swap candidates** — Banked successor orchestrator models: cyankiwi/Qwen3.5-9B-AWQ (true-unification target) and Qwen3-4B-Instruct-2507 (cleanest guided-JSON).  
  *Unblock:* BANKED, not queued: only pursue if a swap is warranted; Qwen3.5-9B is gated on a co-residence fit-test + verbatim-transcription parity, and any probe must run a CHEAP CER/WER pre-check + a <think>-leak stress test FIRST.  
  *Sources:* qwen35_4b_orchestrator_swap_2026_06_01:43/50/51; qwen_migration_research_2026_05_26
- **llama.cpp inference engine at guided-decoding parity** — Adopt llama.cpp's server as a lower-overhead / broader-hardware inference engine.  
  *Unblock:* Revisit only when llama-server reaches parity with vLLM on guided-decoding throughput + stability, or if Apple Silicon becomes first-class.  
  *Sources:* ADR-0001 §Alternatives/Revisit
- **Ollama-compat second-class inference path** — Ship an Ollama-compat inference path as a low-friction trial option.  
  *Unblock:* vLLM is the v1 production engine; Ollama's best-effort JSON can't meet the grammar-constrained requirement. Would-do as a second-class trial path only.  
  *Sources:* ADR-0001 §Alternatives (Ollama)
- **OpenTelemetry GenAI direct instrumentation** — Migrate observability to OpenTelemetry-direct GenAI semantic conventions (one config flip from Langfuse-v4's OTEL backend).  
  *Unblock:* OTel GenAI agent/framework conventions still in Development. Re-check Q4 2026 when conventions stabilize.  
  *Sources:* ADR-0004 §Alternatives/Revisit
- **Bighorn graph store fork** — Switch the embedded graph store to Kineviz's Bighorn Kuzu fork.  
  *Unblock:* Bighorn has no tagged releases yet (can't pin a nonexistent version). Re-evaluate if it ships a stable line and the community converges.  
  *Sources:* ADR-0005 §Alternatives/Revisit
- **e4m3 KV-cache + --calculate-kv-scales as default** — Promote the more-precise fp8_e4m3 KV cache (dynamic scales) over the default e5m2.  
  *Unblock:* A/B was NEUTRAL (exact-match to e5m2 for a small latency cost). Ships as a validated opt-in flag; default stays e5m2 unless e4m3 demonstrably wins. Don't re-research.  
  *Sources:* ADR-0006 §2 amendment; kv_cache_quant_research_2026_05_28; SHIPPED.md:22
- **Qwen3-VL-4B BF16 / better open-weight 4-8B VLM with transformers loader** — Default the doc-VLM to a smaller Qwen3-VL-4B (BF16, no AWQ) or a future 4-8B VLM with materially better OCR + a transformers loader.  
  *Unblock:* The larger model's OCR quality was chosen instead; none better available now. Revisit if AWQ becomes a maintenance burden or a stronger transformers-loadable VLM lands.  
  *Sources:* ADR-0006 §Alternatives/Revisit
- **LanceDB native hybrid search to retire FTS5+RRF** — Use LanceDB's built-in Tantivy FTS + native vector+BM25 hybrid + reranking to retire the parallel SQLite FTS5 store and hand-rolled fusion.py RRF.  
  *Unblock:* The biggest architectural consolidation available, but a RESEARCH+EVAL item: must verify Tantivy matches the tuned unicode61 remove_diacritics 2 multilingual/French behaviour + the chart-block FTS-strip defenses. Not a quick win.  
  *Sources:* db_audit_2026_05_28:35; graph-discovery.md context
- **Qwen3-ASR + ForcedAligner / Whisper-via-vLLM / Parakeet-v3 / Canary-1b-v2 ASR backends** — Deferred ASR-engine alternatives: Qwen3-ASR+ForcedAligner (word/char timestamps), Whisper-via-vLLM (asr_backend=vllm), NVIDIA Parakeet-v3 / Canary-1b-v2; only faster_whisper large-v3-turbo shipped.  
  *Unblock:* turbo already meets the bar. Qwen3-ASR-via-vLLM is DEAD for timestamps (vLLM batch path rejects verbose_json — no segment timestamps). Whisper-via-vLLM lacks word ts (#25750)/VAD + a reported WER blow-up on Ada -> re-weigh when #25750 lands + version-pin + WER spot-check. Parakeet/Canary = banked pilots (12GB long-form friction / NeMo heaviness).  
  *Sources:* ADR-0017 §Considered-Options(2/3/6); audio-asr-route.md §4/§6/§15; asr_audio_scope_2026_06_03; asr_backend.py
- **full mode + structured/grounded long-form output capability tier** — Layer structured/grounded long-form output on full mode's resource posture (vs the removed free-form synthesis).  
  *Unblock:* Partially realized by the structured summarizer (ADR-0008/0010). Revisit when the structured-summary capability ships further. UNCERTAIN whether this is now fully subsumed.  
  *Sources:* ADR-0007 §Expanding/Revisit

## 🚫 Decided-against — NOT pending (39)

_Tried-and-reverted or explicitly rejected. Recorded so they're not re-proposed as open work._

- **P2.5 / GTE-multilingual-base embedder swap** — RUN + CLOSED negative 2026-05-25: GTE-multilingual-base regressed (annual-report 10->8, slide-decks 16->14, french 5->4; -5 ANS net, French worse, zero gains). No embedder beats EmbeddingGemma-300M+native on this stack; no Gemma-4 embedder exists. Code+spec discarded. Narrow revival path: only re-measure the dead FTS-BM25 arm IF a future embedder lands with worse dense recall. (ROADMAP.md:225-227; gemma4_embedder_research; audits/09)
- **Contextual-retrieval LLM context-prefix on embedding input** — TRIED + REVERTED negative: an Anthropic-style ~50-tok situating prefix broke the refusal HARD gate (a real hallucination) + broadly regressed retrieval — the prefix dominates the 300M mean-pooled EmbeddingGemma. Do NOT retry on a small local embedder; the in-distribution lever (native task:/title: prompts) shipped. (ROADMAP.md:151; contextual_retrieval_negative_2026_05_25)
- **FTS BM25-on-NL-questions phrase-wrap fix (lexical-arm activation)** — IMPLEMENTED + VALIDATED -> NEGATIVE/dead lever: the bm25=0 phrase-wrap bug is real but PROVABLY benign (union@50==dense@50 on every corpus; BM25 recall is a strict subset of dense). Reverted with a 'validated benign, do not re-fix' docstring. Revival only via a future embedder swap with worse dense recall — re-run the probe then. (ROADMAP.md:227; audits/09; fts_bm25_nl_scope_2026_05_29)
- **4B-as-doc-VLM / full VLM-role unification** — ATTEMPTED + REVERTED: the unified 4B-VL hallucinated cr350-multidoc (an 8th kill-chain phase) + regressed slide-decks (-3) / handwritten (-2). Kept the dedicated Qwen3-VL-8B; partial unification (4B orchestrator + 8B doc-VLM) is the terminal state. Do NOT retry the 4B as doc-VLM without a stronger vision result. (ADR-0015 §VLM-role; ROADMAP.md:224/397; vlm-vllm-serving.md; qwen35_4b_orchestrator_swap)
- **Perceptual-hash keyframe-OCR dedup** — BUILT then REVERTED as fundamentally UNVIABLE: a whole-frame aHash/dHash can't separate a different slide (~1-2 bit diff) from a held slide under a moving overlay (~17-48 bits) — the dangerous case is closer than the safe one; no Hamming threshold works and the 0.80 floor doesn't contain a false dedup. DON'T retry a whole-frame-hash dedup. (ADR-0018 §Amendment 2026-06-05; companion-merge.md §13-14; companion_levers_2026_06_05; ROADMAP.md:5)
- **LLM transcript-structuring pass (paragraphing / run-on splitting)** — TRIED + VALIDATED NEGATIVE 2026-06-04 + reverted: large-v3-turbo already punctuates, so the 4B returns blocks verbatim (net structuring = ZERO). The real readability lever is coalescing. DON'T retry unless a NON-punctuating ASR enters scope; the adversarially-hardened faithful-transform gate was KEPT as a banked primitive. (ADR-0017 §Decision; audio-asr-route.md §3/§15; transcript_structuring_negative_2026_06_04; core/text.py:911-980)
- **Decompose-and-verify via the SQL stack (text-to-SQL)** — REJECTED as UNSAFE — do NOT reintroduce: because __num is built by the same coerce_number the recompute uses, the aggregate equals the re-sum for every W ('sqlite agrees with sqlite'), and it empirically ships rowid/subquery partial-sums as totals. The safety IS the independent Python row-selection oracle — widen it, never delegate the WHERE to sqlite. (ADR-0014 §Considered-Options(1); table_sql_robustness_2026_05_31:22)
- **OneChart chart-OCR backend** — A/B/C catastrophic failure: every chart figure triggered a CUDA device-side assertion (position-embedding overflow on OOD imagery), 0 useful extractions. Default stays disable_chart_ocr; kept in-tree behind the ADR-0006 carve-out for future chart-heavy-corpus / pinned-older-HF-revision re-attempts (also re-introduces trust_remote_code, deferred pending an ADR amendment). (ROADMAP.md:339-340/655; p33_tracker.md:351; build_status.md)
- **Path B — NeMo Retriever 2-stage chart-OCR** — Deferred: the single-model winner (Nemotron-Parse-v1.2) is sufficient; the 2-stage backend adds no benefit. (ROADMAP.md:294)
- **Browser-side OCR (Tesseract.js / Surya-via-Pyodide / InternVL3-via-ONNX-Web)** — All three rejected as orthogonal/infeasible: Tesseract.js has the same axis-label weakness; Surya-via-Pyodide is blocked (no PyTorch-on-WASM port); InternVL3-9B exceeds ORT-Web's 4GB WASM ceiling. (ROADMAP.md:258-264)
- **SDPA-math deterministic backend for VLM transcription** — Tried + reverted 2026-05-25: CUDA-OOMs on 12 GB (materializes the full attention matrix for ~1k+ visual tokens). best-of-N + cache is the kept determinism mechanism instead. (vlm-transcription-cache.md §Design)
- **ar-12 within-doc cross-SEGMENT relevance-gate conflation** — ATTEMPTED FIX = NEGATIVE (reverted): an evidence-aware scope-attribution criterion didn't catch ar-12 (company gross-margin pinned to the Graphics segment at top_k>=5) AND over-refused a legit segment value. CONFIRMED not LLM-gate-fixable; ACCEPTED as a documented relevance-gate limitation (gate holds at top_k=4). Don't re-attempt the LLM-gate lever. (next_priorities.md:42; eval_nondeterminism_relevance_gate_2026_05_26)
- **Per-class entity-noise regexes / curated entity_stopwords / candidate-noise stopword helper** — Deliberately AVOIDED / removed (bf44f43): brittle per-class regexes (ports/PIDs/FR connectors) + a hand-curated by-name stopword list don't generalise; NO structural metric (degree/df-band/title-overlap) separates CR350 from TCP (statistically identical in a coherent corpus). Root-cause fix is the extractor (OTTER/BERT-NER), not downstream. Keep only the corpus-agnostic shared-docs>=2 floor. (ADR-0011/0012 §Alternatives; ner-enrich.md §Anti-scope; entity_centric_retrieval_2026_05_28; bert_ner_enrich_scope)
- **Entity signal in the /ask retrieve/rerank path** — CONCLUSIVE no headroom at 47 docs (58/58 ANS gold docs already in dense@50) + it would touch the HARD gate. Anti-scope (OTTER is enrich-graph-only, HARD-gate-neutral). Revisit only at large-corpus scale. (graph-discovery.md §Measured-and-NOT-pursued; ner-enrich.md §Anti-scope)
- **Raise OTTER NER threshold to the card's 0.1** — Anti-scope: strangles recall and resurfaces the 'mixed' discovery mirage; the tuned 0.05+union is the live setting. Don't raise without re-measuring. (ner-enrich.md §Anti-scope)
- **Transcript chunk salience / noise-classification layer** — MEASURED -> NEGATIVE: dense retrieval is already a perfect relevance filter (0 social/artifact leaked into substantive top-10); a salience layer adds an LLM pass + HARD-gate risk for ~0 benefit. Key lesson: admin content is NOT noise (students query it). Do NOT build. (audio_video_asr_shipped_2026_06_03:30)
- **W13 near-duplicate SECTION collapse (animation-frame supersets/reorders)** — FP-risky/held: the FP sweep proved every ratio/Jaccard threshold reintroduces parallel-data content loss (different IPs/footnote-numbers/precision-rows under a shared template). The conservative window-1 raw-equality collapse shipped; the richer near-dup collapse needs a heading-equality-gated + code-guarded redesign with its own validation before any retry. (ROADMAP.md:382; audits/10:175; pipeline.py:1185-1187)
- **W17 glyph-spacing / OCR space-join / ref-ID digit-drop repair** — ACCEPTED no-fix: 0 systemic glyph-spacing in the 47-doc vault; the audit accepts residual OCR drift; born-digital docs already use the text layer. (ROADMAP.md:381; audits/10:183; chart_sidecar_2026_06_02)
- **Confidence-weighted discovery ranking (rejected-valence variant)** — NOTE: tracked as data-gated (measured-and-not-shipped, unvalidatable without a should-relate gold set), NOT dead — see the data-gated list. Listed here only to flag that one sweeper categorized it rejected; the more-nuanced data-gated read governs.
- **Confidence-weighted discovery re-weighting by OTTER MENTIONS confidence** — MEASURED and NOT shipped: it measures extraction-TYPICALITY not topical-SPECIFICITY; quality valence is unprovable without a labelled should-relate gold set. (Same item as the data-gated 'Confidence-weighted discovery ranking' — kept there as the governing entry; not dead, just unvalidatable now.) (graph-discovery.md §Measured-and-NOT-pursued)
- **TGI as inference engine** — Close on capability but a smaller small-model/consumer-hardware community; not enough advantage to swim against vLLM ecosystem gravity. Recorded rejection. (ADR-0001 §Alternatives)
- **uv workspace with internal packages** — v1 cost real, benefit zero (no external consumer); a preemptive split bakes in wrong abstractions. Revisit only on team growth / external interest. (ADR-0002 §Alternatives/Revisit)
- **Qwen3-Reranker-0.6B as the DEFAULT reranker** — P2.1 head-to-head lost clearly (median ANS 4 vs 0; ranks generic-CUDA chunks above the literal-answer chunk). Kept only as an opt-in backend; do NOT make default. (Distinct from the still-pending data-gated per-category re-run and the banked smaller-model measurement.) (ADR-0001 §Candidates; ROADMAP.md:49/272)
- **FP16-everywhere dtype policy** — Disqualified by EmbeddingGemma activation-overflow; mixing FP16/BF16 is worst-of-both. Recorded rejection in favour of an explicit bf16 pin. (ADR-0006 §Alternatives)
- **torch_dtype=auto convenience** — Implicit per-model defaults are uneven across the stack (Gemma->FP16 wrong). Rejected for the explicit bf16 pin. (ADR-0006 §Alternatives)
- **DocuFlow skip-failing-tests curation strategy** — Cautionary no-action: the inverse of Memex's HARD gates; vindicates the measure-first discipline. Recorded as a lesson, not pending. (ROADMAP.md:256)
- **LLM resolver / synonym model for artifact scope** — Rejected by the determinism mandate — the prior LLM source-scope clause (#256) was built, validated ineffective, and reverted; the deterministic regex+BM25 resolver replaced it. (artifact-scope.md §Anti-scope)
- **Native pptx renderer (vs Office->PDF)** — Anti-scope: no robust pure-Python pptx->image renderer exists; LibreOffice is the dependency (errors clearly if absent). (office-pdf-conversion.md §Anti-scope)
- **Companion-merge chunk-fusion (slide+commentary in one chunk)** — REJECTED for grounding safety: chunk_id churn + mixed-source text flowing into grounding. The additive per-chunk-pure augmentation shipped instead; a faithfulness-gated 'contextualized' view is a separate gate-sensitive arc. (companion-merge.md §7/§13)
- **Matryoshka truncate_dim change / new embedding task types** — Explicitly out of scope: the embedding dimension stays 768; 'search result' (retrieval) is the correct EmbeddingGemma task type for RAG (no QA/fact-checking task types in v1). (embedding-native-prompts.md §Anti-scope)
- **Move entity extraction onto the answer path** — Anti-scope: would break the HARD-gate-neutral premise; OTTER is enrich-graph-only. (ner-enrich.md §Anti-scope)
- **Table transposition / cell-level parsing** — Anti-scope for table-chunking: row-group + header repetition only. (table-chunking.md §Anti-scope)
- **Prose-label re-attach for degenerate (header-detached) tables** — Deliberately gated OFF: the 10-K segment table (column labels emitted as a detached heading + stray line) stays a documented parse-degeneracy outlier; the only clean fix is fragile. v3 split_merged_columns is distinct (recovers a column-MERGE under a valid bold header). (table-rag.md / table-sql.md §Anti-scope)
- **Cross-document SQL (vs per-doc store)** — Anti-scope for table-sql: per-doc store; the agent queries only docs whose chunks retrieved. (The cross-document corpus-wide variant is tracked as a feature-backlog lever, gated by the recompute fabrication boundary.) (table-sql.md §Anti-scope)
- **LLM-emitted wikilinks** — Anti-scope: wikilinks are deterministic from cited chunks only — no hallucination surface. (wikilink-emission.md §Anti-scope)
- **Shared/multi-user scope sets + 'cite everything in the scope' answer mode** — Anti-scope: single-user local-first (one scope-set file per vault); and the agent always grounds in the retrieved-and-reranked subset of a scoped pool, never 'cites everything'. (scope-sets.md §Anti-scope)
- **Dedicated handwriting HTR model** — Out of scope: the VLM suffices for scanned/handwritten docs; revisit only if accuracy gaps show. (scan-vlm-parse.md §Out-of-scope)
- **H4 deck topic-grouping summarizer knob + embedding-based semantic dedup** — H4 DEFERRED + known-infeasible on noisy-heading decks (deck heading_path is junk; needs robust normalization); embedding-based semantic dedup REJECTED (non-deterministic, degrades under VRAM pressure). Residual purely-semantic cross-paragraph overlap accepted with no clean deterministic fix. (deck_granularity_tracker:68/89/92)
- **Table-under-ranking / margin-bounded table promotion (as a global retrofit)** — NOTE: tracked as a data-gated lever (declined-for-now, evidence-gated), NOT dead — see the data-gated 'cross_encoder under-ranking' entry. Listed here only because one sweeper marked it rejected; declined as a risky global retrofit but revivable with evidence it helps without regressing prose corpora.

## Excluded — deferred-but-since-shipped (17)

_Surfaced by the sweep but already shipped; kept for traceability, not pending._

- P2.1 reranker infra + 1-doc verdict (cross_encoder won; qwen3 opt-in) — the SHIPPED part; the scaled per-category re-run remains pending and is listed separately
- Re-ingest pre-existing PyMuPDF slide-deck content for Tier-0.5 routing (likely covered by the 2026-05-31 batched vault re-process)
- VISION grounding/no-hallucination carve-out for expert mode (noted DONE 2026-06-01)
- Live in-UI co-residence mode hot-switch (subsumed by the dynamic VRAM manager auto mode + /resources panel, 2026-06-04)
- Step 1b chart-extracted off the .md into a manifest sidecar (#362 SHIPPED 2026-06-02)
- Bridge present-as-answer name-only over-grounding (audit-11; /ask backstop + bridge isolated re-verification + name-only consolidation SHIPPED 2026-06-03)
- extract_claims@v1 bridge extractor under-coverage (FIXED as extract_claims/v2, 2026-06-02)
- Citation-grade deck page-map foundation (WIRED + validated + MERGED 2026-06-05; only the per-doc re-parse migration remains, listed separately)
- Companion-merge MaViLS asymmetric-jump DP + start_s prior — the CODE (SHIPPED opt-in 2026-06-05; the corpus-win measurement/gold-set/default-on decision remains, listed separately)
- Summarizer figure-salience table selection (_rank_tables/_table_salience SHIPPED 2026-05-27; the code comment is stale)
- P4.1 wikilink section anchors + P4.4 dynamic VRAM manager (both SHIPPED)
- VLM source-image markdown-link strip (_strip_image_links shipped; the comment is a rationale, not a deferral)
- UI-ingestion v1 hardening backlog Inc 1-7 (B7/B8/B11/B12/B18 etc. MERGED to main 587edaa; only chunk_count gate / half-doc resume / multi-upload queue survive, listed separately)
- Stack-currency eval-gated swaps — PyMuPDF pre-filter (P1.1) + reranker backend SHIPPED (Granite-8B survives as P2.2)
- prompt_tag auto-derive — version-drift class KILLED 2026-06-06 (`prompts/loader.py::prompt_tag_for`/`active_version` derive the tag from the loaded spec; all 18 producer sites converted + a source-scan permanence guard test; branch `feat/prompt-tag-auto-derive`)
- Eval runner per-query error handling + verify ungrounded_reasons overflow — FIXED 2026-06-06 (branch `fix/eval-runner-verify-overflow`): root cause was the guided-decode backend NOT enforcing string max_length → a rambling verify reason truncated the JSON. Fix: the verifier's guided-decode schema is now the reason-less `VerifyIndices` (reasons code-generated by the backstops); the verify node + run_eval fail-closed (distinct `EvalReport.error_count` bucket). slide-decks `memex eval` now completes (error 0); refusal_cf=1.0 + eval-summary held. This UNBLOCKED the slide-decks answer-eval baseline (the W6 residual).
- W6 gold-anchor re-baseline (7 corpora) — DONE 2026-06-06 (branch `eval/w6-gold-anchor-rebaseline`): 5 anchor corpora re-resolved (recall 0.205→0.898) + slide-decks/chat-multiturn re-labeled via an adversarial agent workflow (held-doc principle: only genuinely-stale gold changed); `_baseline_2026_06_06_w6_reanchor` recorded; refusal_cf held 1.0 on all 5 measurable answer corpora; chat mean_recall 1.0. ONE residual carried to the eval-runner item: slide-decks answer-eval still BLOCKED by the ungrounded_reasons-overflow crash (recall-only baseline there).

## Uncertain — implemented-vs-pending unclear (3) — ✅ ALL RESOLVED 2026-06-07 (audit wf_76f65abf)

_Per-line code check DONE — none is genuinely pending. Kept for traceability._

- full mode + structured/grounded long-form output capability tier (ADR-0007) — ✅ **SUBSUMED.** `agents/synthesize.py` is removed (ADR-0009); `full` mode is a deeper-retrieval posture (`retrieval_top_k=18` / `max_model_len=24576`, `core/resources.py`), and the mode-independent ADR-0008 structured summarizer (`agents/document_summarizer.py`) IS the grounded long-form path. No distinct pending capability.
- UI-ingestion residual hardening deferrals (chunk_count searchable gate, half-doc resume/sweep B19) — ✅ **SHIPPED** (test-pinned): the `chunk_count==0` browsable-not-searchable gate (`webui/app.py` + `test_webui.py`) and the `_scan_half_docs` detect+log sweep both ship; the only un-built piece — full half-doc AUTO-resume — was DELIBERATELY scoped out as risky (a heuristic mis-fire must not delete a real doc), not left actionable. The manual `memex index <doc>` path exists.
- Audit-00 X1 phase-N doc-drift sweep — ✅ **RESOLVED** in prior doc syncs: `_PARSER_VERSION=memex.parse@v1` (no `phase-N` strings survive in the named `__init__.py`/cli/registry). The lone residual was a stale `models/registry.py` "Qwen3-8B" docstring (post-ADR-0015) — **FIXED 2026-06-07** (→ the configured 4B).
