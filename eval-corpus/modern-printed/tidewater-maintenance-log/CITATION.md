# Citation

**Synthetic document** authored from scratch for the Memex eval corpus.
No third-party source — the content is original and fictional (a
lighthouse maintenance log). This keeps the ground truth **independent
of the system under test** and **copyright-clean**, per
`docs/eval-corpus-plan.md` (the sanctioned "synthetic" path).

- **License:** Apache-2.0 (same as Memex).
- **Author:** Memex eval corpus.
- **Ground truth:** `ground-truth.md` is the canonical document; it
  predates and is independent of any Memex parse.
- **Source:** `source.pdf` is a faithful rendering of that content via
  `generate.py` (PyMuPDF Story) — a mechanical transform, not Memex.
