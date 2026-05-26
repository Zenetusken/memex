"""Compact-schema retry in the enrich pipeline (`_extract_with_fallback`).

A dense chunk can truncate the full 24-item entity/citation extraction past
the token budget — `complete_structured` then raises `ModelCallError` ("did
not match the requested schema"). Because the 6144 model-len already bounds
prompt+completion, the cure is a one-shot retry with a half-cap (12-item)
compact schema rather than dropping the chunk's enrichment. These tests fake
`complete_structured` to fail on the full schema and succeed on the compact.
"""

from __future__ import annotations

import pytest

from memex.core.errors import ModelCallError
from memex.enrich import pipeline as P
from memex.enrich.entities import EntityList, EntityListCompact, ExtractedEntity


async def test_compact_retry_recovers_a_truncated_full_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[type] = []

    async def _fake(
        *, prompt: str, schema: type, prompt_tag: str, max_tokens: int, **_kw: object
    ) -> tuple[object, int]:
        calls.append(schema)
        if schema is EntityList:  # the full schema "truncates"
            raise ModelCallError("Model output did not match the requested schema.", context={})
        return (
            EntityListCompact(
                entities=[
                    ExtractedEntity(
                        name="Fa0/21", kind="concept", confidence="high", span_text="trunk"
                    )
                ]
            ),
            50,
        )

    monkeypatch.setattr(P, "complete_structured", _fake)

    result = await P._extract_with_fallback(
        prompt="[p]",
        full_schema=EntityList,
        compact_schema=EntityListCompact,
        prompt_tag="extract_entities@v2",
    )
    assert calls == [EntityList, EntityListCompact]  # full failed → compact retry
    assert isinstance(result, EntityList)  # compact subclasses EntityList
    assert result.entities[0].name == "Fa0/21"  # the retry's output is returned


async def test_no_retry_when_full_extraction_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[type] = []

    async def _fake(
        *, prompt: str, schema: type, prompt_tag: str, max_tokens: int, **_kw: object
    ) -> tuple[object, int]:
        calls.append(schema)
        return EntityList(entities=[]), 10

    monkeypatch.setattr(P, "complete_structured", _fake)

    result = await P._extract_with_fallback(
        prompt="[p]",
        full_schema=EntityList,
        compact_schema=EntityListCompact,
        prompt_tag="extract_entities@v2",
    )
    assert calls == [EntityList]  # full succeeded → no compact retry
    assert isinstance(result, EntityList)


async def test_second_failure_propagates_to_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """If even the compact retry fails, the error propagates so the caller's
    gather logs `enrich.chunk_failed` (the chunk is still indexed/searchable).
    """

    async def _fake(
        *, prompt: str, schema: type, prompt_tag: str, max_tokens: int, **_kw: object
    ) -> tuple[object, int]:
        raise ModelCallError("still truncated", context={})

    monkeypatch.setattr(P, "complete_structured", _fake)

    with pytest.raises(ModelCallError):
        await P._extract_with_fallback(
            prompt="[p]",
            full_schema=EntityList,
            compact_schema=EntityListCompact,
            prompt_tag="extract_entities@v2",
        )
