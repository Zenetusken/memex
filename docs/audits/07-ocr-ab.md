# OCR off vs on: empirical A/B (2026-05-20)

Settles the OCR-default question against a real workload, not a
hypothesis. Subject: 109-page NVIDIA GTC slide deck (`S62400 — CUDA
New Features and Beyond`), text-rich PDF with native text layer.
Rig: RTX 4070, 12 GB. Orchestrator: Qwen3-8B-AWQ via vLLM 0.21.

## Setup

| | OCR off | OCR on |
|---|---|---|
| Docling flag | `do_ocr=False` | `do_ocr=True` (RapidOCR per page) |
| Parse time | **96 s** | **1027 s (17.1 min)** |
| Markdown size | 40 160 B | 42 667 B  (+6.2%) |
| Content words | 5054 | 5376  (+322 / +6.4%) |
| Chunks after dedupe | 146 | 157 |
| Indexable chunks | 163 | 178 |
| Swap pressure peak | 4.7 GB | 8.9 GB |

Both runs used identical agent settings (5 chunks expand_graph,
2 chunks per neighbour, regen budget 2). The 4 queries below
hit the same indexed corpus minutes apart — no model warm/cold
differences.

## Results

| Query | OCR off | OCR on | Chunk-set overlap | Claim diff |
|---|---|---|---|---|
| Q1 — *"What is the deck about?"* | ✅ ANSWERED · 5 claims · 5077 tok | ✅ ANSWERED · 5 claims · 5077 tok | **5 / 5** | **bit-identical claims** |
| Q2 — *"What floating-point types?"* | ⚠️ REFUSED · couldn't ground 1 claim | ⚠️ REFUSED · couldn't ground 1 claim | **10 / 10** | — |
| Q3 — *"Energy ↔ data centers?"* | ⚠️ REFUSED · "no explicit link" | ⚠️ REFUSED · "no explicit link" | 9 / 10 | — |
| Q4 — *"What about Kubernetes?"* | ⚠️ correctly REFUSED · "not mentioned" | ⚠️ correctly REFUSED · "not mentioned" | 6 / 10 | — |

**Zero query-outcome changes.** OCR on did not flip a single
refusal to an answer. Where retrieved chunks differed (Q3, Q4),
the differences were lower-ranked chunks that the reranker
correctly deprioritised — they didn't survive into the agent's
final reasoning set.

## Why OCR didn't help

The +322 words OCR added to this deck were:

- Trademark glyphs (`®`, `R`) on slides with NVIDIA logos
- Axis labels and tick numbers on charts (disconnected from
  surrounding context, useless for retrieval)
- Diagram annotations like `Memory`, `Cache`, `Core` floating
  without sentence structure
- The same words that were *already* in the native text layer
  for any slide that had selectable text

The deck's actual content — slide titles, bullet text, tables,
speaker name — lives entirely in the PDF text layer. Docling
extracts that cleanly without OCR. RapidOCR's contribution was
ranking noise.

## When OCR-on is worth it

OCR-on remains the right setting for:

- **Scanned papers** (born-on-paper, no text layer at all)
- **Photographs of whiteboards / printed notes**
- **Screenshots without selectable text**
- **PowerPoint exports where slides were rasterised to images
  instead of preserving the text layer** (rare with modern Office)
- **Old academic papers** (pre-2010, sometimes scanned)

For modern slide decks, papers exported from LaTeX/Word, and
born-digital reports — leave OCR off.

## Recommendation

Keep `MEMEX_PARSE_DOCLING_OCR=0` as the default (commit `fc47fd7`).
Document the per-doc opt-in flag in the parse module's docstring
for users with scanned content. The 10.8× time penalty + the
host-RAM swap pressure make OCR-on a deliberate choice, not a
sensible default.

## Reproducibility

```sh
# OCR off (default — verified canonical)
MEMEX_PARSE_DOCLING_OCR=0  uv run memex parse <doc_id>   # ~96 s

# OCR on (opt-in for scanned content)
MEMEX_PARSE_DOCLING_OCR=1  uv run memex parse <doc_id>   # ~17 min on 109 pages
```

All 4 query JSONs from both runs preserved at
`/tmp/abtest/ocr-{off,on}-q{1..4}.json` during the test session
(disposable; the conclusion above is what survives).
