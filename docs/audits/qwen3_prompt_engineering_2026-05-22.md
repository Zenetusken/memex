# Qwen3 prompt-engineering research notes (2026-05-22)

Research dump from the prompt-engineering investigation following the
v3 prompt instability. Captured here for future reference; not all
items are actioned in this commit.

## Context

While shipping `answer/v3.md` (language-mirror directive for
multilingual support), we hit a HARD GATE failure: placing the
language rule at the top of the prompt caused `refusal_cf` to drop
from 1.0 to 0.77 on the English CUDA-deck eval. Moving the rule to
the END with an explicit "this is a presentation rule; it does NOT
relax the rules above" hedge restored `refusal_cf=1.0`. This
prompted a deeper look at Qwen3-8B-AWQ prompt-engineering best
practices.

## Findings (8 sections)

### 1. Instruction positioning — recency vs primacy

Published literature (IntuitionLabs 2025 synthesis; Hämäläinen 2025
arxiv 2406.15981 on serial position effects in causal-LM attention)
favors **recency for hard constraints**. Items at the start are most
reliably remembered (primacy), items at the very end are second-most
(recency), and middle content tends to be lost.

**Implication for v3.md:** the position fix was consistent with this.
- Hard constraints (literal-presence, no-substitute) at the **top**
- Worked examples in the middle
- Presentation-style rules (language-mirror) at the **end**, with a
  hedge that they don't relax the constraints above

### 2. Determinism and sampling

Qwen team's official non-thinking-mode recommendation: `temperature=0.7,
top_p=0.8, top_k=20, min_p=0`. They explicitly warn that
`presence_penalty > 1.5` on quantized (AWQ/GPTQ) models can trigger
**language mixing** — directly relevant to multilingual content.

Our `models/client.py` defaults to `temperature=0.0` which is good for
eval reproducibility but conflicts with Qwen's recommendation.

**Recommendation** (not actioned in this commit):
- Eval mode: keep `temperature=0.0` + add explicit `seed=42`
- Production mode: `temperature=0.1, top_p=0.8, min_p=0,
  presence_penalty=1.0` (not the team's 1.5, which causes language
  mixing on AWQ)

### 3. System vs user message split

Qwen3 ships without a default system prompt and per HF deep-dive blog
does NOT strongly distinguish system vs user roles for instruction-
following — it follows the most recent instruction regardless of
role. ChatML honors `<|im_start|>system` but it's not a hard-
constraint amplifier the way it is on Llama-3 or GPT-4.

Our prompts have `role: user` (whole prompt body in one user message).

**Recommendation** (not actioned in this commit):
- Split into system (role + rules + examples + schema) and user
  (query + chunks + optional feedback)
- Benefit is mostly architectural cleanliness + vLLM prefix-cache
  reuse on the static system block, not a quality lift

### 4. xgrammar specifics

xgrammar enforces JSON structure at the decoding layer. Our `Reply
ONLY with a JSON object matching this schema:` sentence is
**redundant for correctness** but useful for steering — it nudges
the model toward valid JSON faster (fewer tokens spent fighting the
grammar mask), which matters for our `max_tokens=640` budget.

The schema block in the prompt body is also redundant when
`response_format` is set. Removing it would free ~80 tokens.
**Trade-off**: keeping it in-prompt helps the model "see" the field
names earlier in attention, which empirically improves field-naming
accuracy on small models. Leave as-is for now.

### 5. Worked-example positioning

Few-shot literature (arxiv 2305.11383, 2509.13196 "Few-shot dilemma")
supports **"tell, then show"** for instruction-tuned models: abstract
rule first, concrete examples second. v3.md already does this.

However, our worked examples are **all negative** ("what NOT to do").
Recent guidance flags that all-negative few-shot can cause models to
fixate on the negative pattern and refuse even when they shouldn't.

**Actioned in this commit:** add one positive example to v3.md
before the three negatives.

### 6. Tool use / structured output cleanliness

Qwen3 is heavily trained for function-calling and excels at JSON.
With xgrammar enforcing schema and `max_length=N` bounds on each
emit field, our setup is configuration-correct. No prompt change
needed.

### 7. Anti-patterns to avoid

- **Conflicting instructions across positions** — our bug was a
  textbook case. The "language-mirror" rule at top read as a hard
  constraint, conflicting with the no-substitute rule.
- **Don't mix `/think` and `/no_think` tags in the prompt body** —
  they override anything else. We don't use these.
- **Don't use `presence_penalty > 1.5` on multilingual prompts** —
  triggers language mixing on quantized models. Affects future
  sampling tuning.
- **Don't include thinking content in conversation history** —
  irrelevant to single-turn v3.md but relevant if we ever go
  multi-turn for feedback loops.

### 8. Multilingual prompting

Qwen3 is trained on 119 languages with strong cross-lingual transfer.
**No documented "best practice" pattern exists for English-prompt +
non-English-output**, but Qwen-MT internally uses the explicit
"output language matches query language" instruction pattern. Our
v3.md follows this.

Quantized models are more prone to language mixing under high
`presence_penalty`. Keep ≤1.0 if/when we tune sampling.

## Action items

| # | Item | Effort | Priority |
|---|---|---|---|
| 1 | Add one positive worked example to v3.md | Low | High (actioned here) |
| 2 | Add `seed=42` to eval-mode `complete_structured` calls | Low | Medium |
| 3 | Tune production sampling (`temperature=0.1, top_p=0.8, min_p=0, presence_penalty=1.0`) | Medium | Medium |
| 4 | Split v3.md into system + user messages | Medium | Low (architectural cleanup) |
| 5 | Trim redundant schema block from prompt body | Low | Low (saves ~80 tokens) |

Items 2-5 deferred to future work; this commit only addresses item 1
(the positive-example fix).

## Sources

- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3-8B-AWQ model card](https://huggingface.co/Qwen/Qwen3-8B-AWQ)
- [Qwen-3 Chat Template Deep Dive (HF blog)](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
- [LLM Position Bias: Primacy and Recency Effects in Prompts (IntuitionLabs)](https://intuitionlabs.ai/articles/llm-position-bias-primacy-recency-effects)
- [Serial Position Effects of LLMs (arxiv 2406.15981)](https://arxiv.org/abs/2406.15981)
- [The Few-shot Dilemma (arxiv 2509.13196)](https://arxiv.org/pdf/2509.13196)
- [Instability of Safety: Random Seeds and Temperature (arxiv 2512.12066)](https://arxiv.org/pdf/2512.12066)
- [Cross-Lingual Prompt Steerability (arxiv 2512.02841)](https://arxiv.org/pdf/2512.02841)
- [XGrammar paper (arxiv 2411.15100)](https://arxiv.org/pdf/2411.15100)
- [Qwen3 blog (Qwen team)](https://qwenlm.github.io/blog/qwen3/)
