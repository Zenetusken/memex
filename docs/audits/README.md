# Audit reports

Standing record of correctness/security/wiring audits run against the
codebase. Each audit is captured here so that the "why" behind a class
of fixes survives long after the commits land.

| Date | Audit | Method |
|---|---|---|
| 2026-05-19 | Stack currency | single agent — see `memory/stack_currency_audit.md` (off-tree, time-sensitive) |
| 2026-05-20 | CUDA dispatch | four parallel agents → ADR-0006 |
| 2026-05-20 | Multi-agent bug-hunt | four parallel agents (resource/concurrency, error/edge, wiring/signatures, quality) — see `00-synthesis.md` through `04-wiring.md` |
| 2026-05-20 | E2E + load test on RTX 4070 | live verification of every audit fix — see `05-e2e-loadtest.md` |
| 2026-05-20 | 8B-AWQ load test on RTX 4070 | post-tuning verification of the production-target orchestrator — see `06-8b-loadtest.md` |
| 2026-05-20 | OCR off vs on (109-page slide deck) | empirical A/B settling the parse-default question on born-digital PDFs — see `07-ocr-ab.md` |
| 2026-05-22 | Qwen3 prompt engineering | research synthesis on prompting techniques for Qwen3-8B-AWQ at our scale; 5 follow-ups shipped (sampling, system/user split, schema trim, settings centralization, schema tightening) — see `qwen3_prompt_engineering_2026-05-22.md` |
| 2026-05-22 | MIRACL-fr retrieval benchmark | nDCG@10 = 0.807 with cross_encoder rerank vs 0.755 dense-only; validates the bf16 + cross_encoder pipeline composition for French — see `miracl_fr_2026-05-22.md` |
| 2026-05-23 | Chart-OCR landscape (P3.3-c) | three parallel research agents — OneChart root cause + NVIDIA offerings + 2025-26 chart-OCR landscape. Donut/VisionEncoderDecoder identified as the architectural-safe family — see `chart_ocr_landscape_2026-05-23.md` |
| 2026-05-23 | Chart-OCR fine-tuning research | three parallel agents on official fine-tuning workflows for UniChart / NeMo Retriever / Nemotron-Parse — see `chart_ocr_finetune_research_2026-05-23.md` |
| 2026-05-23 | Chart-OCR backend shootout (P3.3-c) | A/B/C/D eval of DePlot / UniChart / OneChart / Nemotron-Parse-v1.2 against the slide-decks corpus. Nemotron-Parse wins (no prose regression). Late-session v7 banner added after the chunker / converter / prompt fixes — see `chart_ocr_shootout_2026-05-23.md` |
| 2026-05-23 | OneChart retry (P3.3-b) | A/B/C eval revealed CUDA device-side assertion on every chart figure (OPT decoder positional-embedding overflow on OOD imagery); default stays DePlot-only — see `onechart_2026-05-23.md` |
| 2026-05-23 | Post-v7 verification audit | four parallel agents (resource/concurrency, error/edge, wiring, quality) fanned out on the v7 fix arc; 6 critical/important findings fixed inline (truncated-chart-block defense, multicolumn nested-brace fix, label-number heuristic tightening, ungrounded_reasons overflow log, chunker N+1 regex elimination, stale-docstring fix); 21 new tests; 255 → 276 passing; no eval regression — see `post_v7_2026-05-23.md` |

## Pattern

When you run a new audit, follow the 2026-05-20 template:

1. **Fan out** to specialist subagents, each with a focused lens.
2. **Synthesise** their findings into a deduplicated priority list (the `00-synthesis.md` file).
3. **Fix in batches** ordered by severity (security → broken-now → data-loss → correctness → drift).
4. **Verify live** on the reference rig — every fix should have an observable signal that confirms it landed (`05-e2e-loadtest.md` is the model).

The four-agent pattern is the user-approved standard. New patterns are fine; document them here when you try one.

## On what NOT to commit here

- **In-progress findings** (the raw subagent output before synthesis).
- **Audits whose conclusions never made it into code** — file an issue or an ADR if they're worth keeping; this directory is for the record of *what happened*, not what should happen.
- **Time-sensitive observations that decay quickly** (e.g., "PyPI is rate-limiting today"). Those go in memory under `~/.claude/projects/.../memory/`, not in-tree.
