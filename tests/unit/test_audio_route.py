"""Unit tests for the audio parse route helpers (Increment 4, ADR-0017): the pure
`_assemble_transcript` / `_format_timestamp` and the chunker time-attribution.
"""

from __future__ import annotations

from pathlib import Path

from memex.index.chunker import _time_range_for_offset, chunk_document
from memex.parse.asr_backend import ASRSegment
from memex.parse.pipeline import _assemble_transcript, _format_timestamp
from memex.vault.store import DocumentRef, Frontmatter, VaultDocument


def _doc(body: str) -> VaultDocument:
    ref = DocumentRef(
        doc_id="aud-test",
        markdown_path=Path("/tmp/aud-test.md"),  # noqa: S108 — test fixture
        asset_dir=Path("/tmp/aud-test"),  # noqa: S108
        source_path=None,
        content_sha256="0" * 64,
    )
    return VaultDocument(ref=ref, frontmatter=Frontmatter(title="lecture"), body=body, mtime_ns=0)


def test_format_timestamp() -> None:
    assert _format_timestamp(0) == "00:00"
    assert _format_timestamp(62) == "01:02"
    assert _format_timestamp(3661) == "01:01:01"
    assert _format_timestamp(-5) == "00:00"  # clamps negatives


def test_assemble_transcript_body_and_block_spans() -> None:
    segs = [
        ASRSegment(text="hello world", start_s=0.0, end_s=2.0, language="en"),
        ASRSegment(text="bonjour", start_s=62.0, end_s=64.0, language="fr"),
    ]
    body, tsegs = _assemble_transcript(segs)
    assert body == "## [00:00]\nhello world\n\n## [01:02]\nbonjour"
    # each segment's char-span addresses its whole BLOCK (header + text), EXACTLY.
    assert body[tsegs[0].char_start : tsegs[0].char_end] == "## [00:00]\nhello world"
    assert body[tsegs[1].char_start : tsegs[1].char_end] == "## [01:02]\nbonjour"
    assert [t.index for t in tsegs] == [0, 1]
    assert tsegs[1].start_s == 62.0 and tsegs[1].end_s == 64.0 and tsegs[1].language == "fr"


def test_assemble_transcript_empty() -> None:
    body, tsegs = _assemble_transcript([])
    assert body == ""
    assert tsegs == []


def test_time_range_for_offset() -> None:
    intervals = [(0, 22, 0.0, 3.0), (24, 40, 62.0, 66.0)]
    assert _time_range_for_offset(intervals, 0) == (0.0, 3.0)
    assert _time_range_for_offset(intervals, 30) == (62.0, 66.0)
    assert _time_range_for_offset(intervals, 100) == (62.0, 66.0)  # past last → last (drift)
    assert _time_range_for_offset([], 0) is None


def test_chunk_document_time_range_from_segments() -> None:
    segs = [
        ASRSegment(
            text="the router forwards packets and frames across the link", start_s=0.0, end_s=4.0
        ),
        ASRSegment(
            text="VLANs segment the broadcast domain into isolated parts", start_s=62.0, end_s=68.0
        ),
    ]
    body, tsegs = _assemble_transcript(segs)
    intervals = [(t.char_start, t.char_end, t.start_s, t.end_s) for t in tsegs]
    chunks = chunk_document(_doc(body), segment_intervals=intervals)
    assert chunks
    assert all(c.time_range is not None for c in chunks)  # every transcript chunk is time-anchored
    assert chunks[0].time_range == (0.0, 4.0)  # the first chunk is the first segment's block


def test_chunk_document_without_segments_leaves_time_range_none() -> None:
    chunks = chunk_document(_doc("## [00:00]\nsome transcript text for chunking here"))
    assert chunks
    assert all(
        c.time_range is None for c in chunks
    )  # non-audio / no intervals → None (back-compat)
