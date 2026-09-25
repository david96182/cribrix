"""Unit tests for Stage 3 — context triage (the cribrix)."""

from __future__ import annotations

import asyncio
import time

from cribrix.clients.jev import FakeJevClient, PassageAssessment
from cribrix.pipeline.triage import triage_chunks
from cribrix.schemas import Chunk
from tests.conftest import ScriptedJev


def _jev(
    table: dict[str, tuple[float, float, float]], *, fail: set[str] | None = None
) -> ScriptedJev:
    """content -> (relevance, answers, injection)."""

    def assess(_q: str, passage: str) -> PassageAssessment:
        rel, ans, inj = table.get(passage, (0.0, 0.0, 0.0))
        return PassageAssessment(rel, 1.0, ans, inj)

    return ScriptedJev(assess=assess, fail=fail)


def _chunk(cid: int, content: str) -> Chunk:
    return Chunk(id=cid, document_id="d", content=content)


async def test_keeps_only_chunks_at_or_above_threshold() -> None:
    """The threshold is inclusive at the boundary, and scores are continuous."""
    items = [_chunk(1, "high"), _chunk(2, "exact"), _chunk(3, "low")]
    jev = _jev({"high": (0.9, 1, 0), "exact": (0.7, 1, 0), "low": (0.69, 1, 0)})

    scored, kept = await triage_chunks(jev, "q", items, threshold=0.7, max_keep=10)

    assert {c.id for c in kept} == {1, 2}
    assert len(scored) == 3
    assert next(s for s in scored if s.chunk.id == 3).drop_reason == "irrelevant"


async def test_relevant_but_non_answering_chunks_are_dropped() -> None:
    """Relevance is not answerability — the 'early termination penalty' case."""
    items = [_chunk(1, "refund window"), _chunk(2, "penalty clause")]
    jev = _jev({"refund window": (0.7, 0.1, 0), "penalty clause": (0.9, 0.9, 0)})

    scored, kept = await triage_chunks(
        jev, "q", items, threshold=0.5, evidence_threshold=0.5, max_keep=10
    )

    assert [c.id for c in kept] == [2]
    assert next(s for s in scored if s.chunk.id == 1).drop_reason == "does_not_answer"


async def test_prompt_injection_is_dropped_even_when_relevant() -> None:
    """Security is checked first: a relevant, 'answering' injection never passes."""
    items = [_chunk(1, "forum post"), _chunk(2, "docs")]
    jev = _jev({"forum post": (1.0, 1.0, 0.99), "docs": (0.9, 0.9, 0.05)})

    scored, kept = await triage_chunks(
        jev, "q", items, threshold=0.5, injection_max=0.7, max_keep=10
    )

    assert [c.id for c in kept] == [2]
    dropped = next(s for s in scored if s.chunk.id == 1)
    assert dropped.drop_reason == "prompt_injection"
    assert dropped.injection == 0.99


async def test_every_signal_is_recorded_in_the_trace() -> None:
    items = [_chunk(1, "a")]
    scored, _ = await triage_chunks(
        _jev({"a": (0.8, 0.6, 0.1)}), "q", items, threshold=0.5, max_keep=5
    )
    assert (scored[0].relevance, scored[0].answers, scored[0].injection) == (0.8, 0.6, 0.1)


async def test_returns_empty_when_nothing_clears_the_bar() -> None:
    items = [_chunk(1, "a"), _chunk(2, "b")]
    scored, kept = await triage_chunks(
        _jev({"a": (0.3, 1, 0), "b": (0.5, 1, 0)}), "q", items, threshold=0.7, max_keep=10
    )
    assert kept == []
    assert all(not s.kept for s in scored)


async def test_survivors_are_ordered_most_relevant_first() -> None:
    items = [_chunk(1, "mid"), _chunk(2, "top"), _chunk(3, "bottom")]
    jev = _jev({"mid": (0.8, 1, 0), "top": (0.95, 1, 0), "bottom": (0.72, 1, 0)})
    _, kept = await triage_chunks(jev, "q", items, threshold=0.7, max_keep=10)
    assert [c.id for c in kept] == [2, 1, 3]


async def test_max_keep_truncates_to_the_best_chunks() -> None:
    items = [_chunk(i, f"c{i}") for i in range(1, 6)]
    jev = _jev({f"c{i}": (0.70 + i / 100, 1, 0) for i in range(1, 6)})
    scored, kept = await triage_chunks(jev, "q", items, threshold=0.7, max_keep=2)
    assert [c.id for c in kept] == [5, 4]
    assert sum(s.drop_reason == "over_budget" for s in scored) == 3


async def test_empty_input_short_circuits_without_calling_jev() -> None:
    jev = _jev({})
    scored, kept = await triage_chunks(jev, "q", [], threshold=0.7, max_keep=5)
    assert (scored, kept, jev.assessed) == ([], [], [])


async def test_one_request_per_chunk() -> None:
    """All three questions about a chunk travel in a single request."""
    items = [_chunk(i, f"c{i}") for i in range(4)]
    jev = FakeJevClient()
    await triage_chunks(jev, "q", items, threshold=0.0, max_keep=10)
    assert jev.calls["assess_passage"] == 4


async def test_single_failure_does_not_sink_the_request() -> None:
    items = [_chunk(1, "good"), _chunk(2, "broken")]
    jev = _jev({"good": (0.9, 1, 0)}, fail={"broken"})
    scored, kept = await triage_chunks(jev, "q", items, threshold=0.7, max_keep=10)
    assert [c.id for c in kept] == [1]
    assert next(s for s in scored if s.chunk.id == 2).drop_reason == "assessment_failed"


async def test_total_failure_yields_empty_keep_set() -> None:
    """If every request fails we keep nothing, so the caller refuses. Fail-closed."""
    items = [_chunk(1, "a"), _chunk(2, "b")]
    _, kept = await triage_chunks(_jev({}, fail={"assess"}), "q", items, threshold=0.0, max_keep=10)
    assert kept == []


async def test_assessment_is_concurrent_not_sequential() -> None:
    """Ten chunks at 50ms each take ~500ms serially and ~50ms concurrently."""
    items = [_chunk(i, f"chunk {i}") for i in range(10)]
    jev = FakeJevClient(latency_s=0.05)

    start = time.perf_counter()
    await triage_chunks(jev, "q", items, threshold=0.0, max_keep=10, max_concurrency=10)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.25, f"triage appears serialised ({elapsed:.3f}s for 10x50ms)"


async def test_concurrency_limit_is_respected() -> None:
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    class _Tracking(ScriptedJev):
        async def assess_passage(self, question: str, passage: str) -> PassageAssessment:
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            return PassageAssessment(0.9, 1.0, 0.9, 0.0)

    items = [_chunk(i, f"c{i}") for i in range(20)]
    await triage_chunks(_Tracking(), "q", items, threshold=0.7, max_keep=20, max_concurrency=4)
    assert peak <= 4, f"concurrency limit exceeded: peak={peak}"


async def test_every_candidate_appears_in_the_audit_trail() -> None:
    items = [_chunk(i, f"c{i}") for i in range(1, 6)]
    jev = _jev({f"c{i}": (i / 10, 1, 0) for i in range(1, 6)})
    scored, _ = await triage_chunks(jev, "q", items, threshold=0.4, max_keep=10)
    assert {s.chunk.id for s in scored} == {1, 2, 3, 4, 5}
