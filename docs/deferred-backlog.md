# Deferred backlog — items not yet implemented

**Generated:** 2026-06-05 · **Pending items:** 123 (+ 39 explicitly decided-against).

> **What this is.** A point-in-time snapshot of every consciously-deferred item across the project,
> synthesized from a sweep of `docs/ROADMAP.md`, the ADRs, the specs, the agent's memory trackers,
> the audit reports, and code comments — deduplicated, with anything since-shipped removed. It is a
> **regenerable digest, not live state**: an item here may have shipped since; verify against current
> code/ROADMAP before treating any line as fact. The authoritative trackers remain ROADMAP.md (status)
> and the per-feature ADRs/specs (the *why*). Many entries are granular — some are sub-items of a larger arc.

> **Last reconciled 2026-06-05:** dropped the orphaned `duckdb` dep (pruned) and the entire
> "audit-2026-05-20 open items" entry — all four of its sub-items (D2 per-doc `_DOC_LOCKS`, #23 vendored
> Tailwind+HTMX, #27 watcher test driving `_drain_one`, #29 MCP tools return pydantic-not-dict) were
> verified DONE against current code. Net: this audit is fully closed.

## How to read the categories

- **🎯 Queued / next-pickup** — intended soon.
- **⛔ Data-gated / blocked** — needs corpus data, a model/hardware, or corpus scale before it can even be measured.
- **📋 Feature backlog** — would-do, lower priority, no external blocker.
- **🔬 Researched + banked** — has a verdict; waiting on a forcing function.
- **🚫 Decided-against** — tried-and-reverted or rejected; **NOT pending** — listed only so they're not mistaken for open work.

## 🎯 Queued / next-pickup (4)

_Intended soon — the most actionable._

- **VLM prompt decorative-narration suppression (audit-10 W6)** — Calm-register VLM prompt to skip decorative-image / editorial-narration noise (~15-25 vault blocks) in diagram transcription.  
  *Unblock:* Implemented + validated content-safe/gate-safe but FAILED ship bar with a consistent -1 ANS on cr350-diagrams (the 'transcribe every line' preservation guard crowds retrieval). Reverted byte-identical. Own-session: author a less-verbosity-inducing guard validated ANS-neutral across ALL VLM corpora before a vault-wide re-parse. Finding in vlm_backend._PROMPT NB comment.  
  *Sources:* ROADMAP.md:386/15; audits/10:171; vlm_backend.py:76-82; next_priorities.md; build_status.md
- **Docling height-leveling on dense UNNUMBERED docs (step-3b residual)** — Un-flatten the dominant section tier on a dense unnumbered doc (force-docling'd 10-K: 415 real section titles pinned at the H5 cap) via frequency/mode-anchored re-tiering.  
  *Unblock:* The level-cap bounds depth but can't un-flatten; the section-number signal doesn't apply to unnumbered headings; mode-aware re-tiering trades one failure mode for another. Needs a focused session + a multi-doc force-docling A/B. Lower priority (the 10-K's DEFAULT route is PyMuPDF, clean tree).  
  *Sources:* ROADMAP.md:387/15; audits/10:169; next_priorities.md; build_status.md
- **eval-expert MAJORITY-of-N gated substring policy (v1.1 must-fix)** — Switch gated substring cases from any-run-fail to MAJORITY-of-N so a rare REAL affirm surfaces via gate_run_stable, not a flaky fail.  
  *Unblock:* Demonstrated-needed (a 1-in-22 false-fire flipped hard_gates_pass ~21% of runs) but made NON-urgent by the directional-phrasing blocklist fix; documented as the v1.1 refinement. Pickable now.  
  *Sources:* eval_expert_2026_06_01:25-27
- **UI-ingestion residual hardening deferrals (chunk_count 'searchable' gate, half-doc lifespan reconcile, V3c lifecycle)** — B6/B12 chunk_count searchable gate; B16/B19 half-doc (ingest-but-no-index) resume/sweep on startup; V3c supervisor.start retry + chart-OCR pre-flight retrieval unload. Detect+log shipped; the resume/sweep ACTION did not.  
  *Unblock:* Lower-value/larger items deferred from the UI-ingestion livetest + dynamic-VRAM-manager; single-user localhost v1. Note: most of the deferred backlog (B7/B8/B11/B18) MERGED 587edaa — verify each against current code before picking up.  
  *Sources:* app.py:244; ui_ingestion_livetest_2026_06_05; ui_ingestion_deferred_done_2026_06_05; dynamic_vram_manager_2026_06_04:20

## ⛔ Data-gated / blocked (39)

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
- **Citation-chain following (transitive multi-hop CITES traversal)** — Follow CITES Document->Document edges multi-hop (citation_paths()); 1-hop References shipped, transitive chains did not. Pre-registered design + audit harness scripts/citation_graph_audit.py shipped.  
  *Unblock:* DATA-GATED: the graph is a depth-1 star (~6 CITES edges, 0 multi-hop, academic=0). Bar = >=15 edges / >=5 docs / >=1 multi-hop on a curator-supplied 5-10 paper citation-linked cluster; build IFF the bar clears.  
  *Sources:* ROADMAP.md:196; ADR-0011 build-out; graph-discovery.md; next_priorities.md:60; db_audit_2026_05_28; ner_leverage_buildout; graph_store.py:112/735; mcp/server.py:211-275
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
- **Companion alignment gold-set authoring** — Author the transcript->slide gold alignment set that gates τ_null, the DP default-on decision, and the augment default-ON flip.  
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

## 📋 Feature backlog (63)

_Would-do, lower priority — no external blocker, just unscheduled._

- **De-hyphenation markdown cleanup** — Low-risk post-process to rejoin end-of-line-hyphenated words to improve embedding + BM25 token matching (verified no rejoin exists in parse/).  
  *Unblock:* Only worth doing IF parsed output actually shows broken hyphenated words — verify first. Tier 6.  
  *Sources:* ROADMAP.md:248
- **Adaptive batch-size autotune (rerank/VLM OOM-backoff)** — Geometric probe + OOM-backoff to replace the hand-set MEMEX_RERANK_BATCH_SIZE=1 / per-doc VLM batch (mined from DocuFlo).  
  *Unblock:* Tier 6 roadmap; the rerank-OOM batch-1 fallback shipped as a stopgap — this is the durable replacement.  
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
- **download-models.py implementation** — The model-download/cache CLI is a stub that prints 'not yet implemented' and exits 1.  
  *Unblock:* Never implemented; a real impl resolves model ids from MemexSettings.models, downloads via huggingface-cli with hash verification, reports disk usage.  
  *Sources:* scripts/download-models.py:18
- **tiktoken-counted chunk tokens** — Swap the chunker's word-count 'tokens' for real tiktoken-counted tokens (current word-count is ~1.3x lower than real transformer tokens).  
  *Unblock:* On the roadmap as a future swap; couple to the P1.6 chunker-size verdict.  
  *Sources:* core/config.py:558-560
- **SQLite/LanceDB connection reuse (long-lived handles)** — Reuse per-store DB handles to avoid re-opening FTS 3-5x + LanceDB 2-4x per /ask (the SQLite-audit's highest-leverage perf item).  
  *Unblock:* RISKIER refactor: a reused LanceDB handle needs read_consistency_interval=timedelta(0) or it misses the indexer's cross-process writes. A focused refactor + validation, not a safe drive-by.  
  *Sources:* db_audit_2026_05_28:32/17; graph_store.py:48-49; ROADMAP.md:372
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
- **Reasoning-over-retrieved grounded-synthesis (supported-by-evidence-set gate)** — A middle-ground expert variant: multi-hop synthesis over cited chunks with a relaxed 'supported-by-the-evidence-SET' gate vs strict literal grounding.  
  *Unblock:* Surface B shipped as pure model-knowledge reasoning, NOT this grounded-synthesis variant; recalibrating verify_grounding to evidence-set support (the hallucination firewall) is the riskiest part and was left open.  
  *Sources:* reasoning_expert_mode_scope_2026_05_29:24/34
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
- **Summarizer publication-metadata key-point suppression** — Drop boilerplate publication-metadata key-points (NIST 'is titled / authored by / FISMA / contact email') that still lead the headline points.  
  *Unblock:* The v2 MAP nudge cleaned the abstract but the model still extracts some metadata key-points; a prompt nudge alone isn't reliable per-point. Decide deterministic post-filter vs section-salience vs accept with eval evidence.  
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
- **Per-claim wikilinks** — Emit wikilinks per individual claim instead of a flat deduped FinalResponse.wikilinks 'Sources' list.  
  *Unblock:* Anti-scope for v1; a possible later refinement.  
  *Sources:* wikilink-emission.md §Anti-scope
- **Auto-derived / tag-derived scope sets** — Scope sets auto-derived by tag (e.g. 'all SRWE decks') instead of explicit hand-picked selection.  
  *Unblock:* Anti-scope (a set is an explicit hand-picked selection); tag-derived scoping is a separate future idea.  
  *Sources:* scope-sets.md §Anti-scope
- **Multi-upload ingestion QUEUE (webui)** — Queue concurrent browser ingests instead of single-flight rejecting a 2nd ingest with a 409.  
  *Unblock:* v1 non-goal by contract (single-GPU rig runs one ingest at a time): a design change (job model + per-job GPU scheduling), not hardening. Warranted only on multi-user / multi-GPU.  
  *Sources:* ROADMAP.md:6/183-184; ADR-0019 §Negative/Revisit; ui-ingestion.md; ui_ingestion_deferred_done_2026_06_05
- **Answer-graph stale doc-picker scope clearing (UX)** — Clear/surface a leftover doc-picker scope selection that silently scopes the next /ask -> confusing-but-correct refusal.  
  *Unblock:* Minor UX gotcha the user hit; noted as 'consider clearing/surfacing stale scope' but not built.  
  *Sources:* next_priorities.md:74

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

## Excluded — deferred-but-since-shipped (14)

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

## Uncertain — implemented-vs-pending unclear (3)

_Flagged by the sweep; needs a per-line code check before treating as pending._

- full mode + structured/grounded long-form output capability tier (ADR-0007) — partially realized by the structured summarizer ADR-0008/0010; unclear if the remaining 'further' capability is still a distinct pending item or fully subsumed
- UI-ingestion residual hardening deferrals (chunk_count searchable gate, half-doc resume/sweep B19) — detect+log shipped and much of the backlog merged 587edaa; each item needs a per-line code check to confirm still-pending vs shipped
- Audit-00 X1 phase-N doc-drift sweep — a docs/drift item that may be partly stale post-consolidation; verify against current __init__.py / cli / registry / _PARSER_VERSION
