"""Unit tests for Stage 3 — context triage (the cribrix)."""

from __future__ import annotations

import asyncio
import time

from cribrix.clients.jev import FakeJevClient, JevError
from cribrix.pipeline.triage import triage_chunks
from cribrix.schemas import Chunk


class _FixedScoreJev:
    """Jev stub returning a preset score per chunk content."""

    def __init__(self, scores: dict[str, float], *, fail_on: set[str] | None = None) -> None:
        self._scores = scores
        self._fail_on = fail_on or set()
        self.calls = 0

    async def choice(  # pragma: no cover
        self, context: object, options: dict[str, str]
    ) -> tuple[str, float]:
        return next(iter(options)), 1.0

    async def score(self, question: str, document: str) -> float:
        self.calls += 1
        if document in self._fail_on:
            raise JevError("scoring failed")
        return self._scores.get(document, 0.0)

    async def score_batch(self, question: str, documents: list[str]) -> list[float]:
        return [await self.score(question, d) for d in documents]  # pragma: no cover

    async def grounded(self, source: str, claim: str) -> float:  # pragma: no cover
        return 1.0

    async def health(self) -> bool:  # pragma: no cover
        return True

    async def aclose(self) -> None:  # pragma: no cover
        return None


def _chunk(cid: int, content: str) -> Chunk:
    return Chunk(id=cid, document_id="d", content=content)


async def test_keeps_only_chunks_at_or_above_threshold() -> None:
    """The threshold is inclusive at the boundary."""
    items = [_chunk(1, "high"), _chunk(2, "exact"), _chunk(3, "low")]
    jev = _FixedScoreJev({"high": 0.9, "exact": 0.7, "low": 0.69})

    scored, kept = await triage_chunks(jev, "q", items, threshold=0.7, max_keep=10)

    assert {c.id for c in kept} == {1, 2}
    assert len(scored) == 3
    assert {s.chunk.id for s in scored if s.kept} == {1, 2}


async def test_returns_empty_when_nothing_clears_the_bar() -> None:
    """The case the whole project exists for: no chunk is good enough."""
    items = [_chunk(1, "a"), _chunk(2, "b")]
    jev = _FixedScoreJev({"a": 0.3, "b": 0.5})

    scored, kept = await triage_chunks(jev, "q", items, threshold=0.7, max_keep=10)

    assert kept == []
    assert len(scored) == 2
    assert all(not s.kept for s in scored)


async def test_survivors_are_ordered_most_relevant_first() -> None:
    items = [_chunk(1, "mid"), _chunk(2, "top"), _chunk(3, "bottom")]
    jev = _FixedScoreJev({"mid": 0.8, "top": 0.95, "bottom": 0.72})

    _, kept = await triage_chunks(jev, "q", items, threshold=0.7, max_keep=10)

    assert [c.id for c in kept] == [2, 1, 3]


async def test_max_keep_truncates_to_the_best_chunks() -> None:
    """The budget cap must drop the weakest survivors, not arbitrary ones."""
    items = [_chunk(i, f"c{i}") for i in range(1, 6)]
    jev = _FixedScoreJev({f"c{i}": 0.70 + i / 100 for i in range(1, 6)})

    _, kept = await triage_chunks(jev, "q", items, threshold=0.7, max_keep=2)

    assert [c.id for c in kept] == [5, 4]


async def test_empty_input_short_circuits_without_calling_jev() -> None:
    jev = _FixedScoreJev({})
    scored, kept = await triage_chunks(jev, "q", [], threshold=0.7, max_keep=5)
    assert (scored, kept, jev.calls) == ([], [], 0)


async def test_single_scoring_failure_does_not_sink_the_request() -> None:
    """A failed score degrades that chunk to 0.0; the others still count."""
    items = [_chunk(1, "good"), _chunk(2, "broken")]
    jev = _FixedScoreJev({"good": 0.9}, fail_on={"broken"})

    scored, kept = await triage_chunks(jev, "q", items, threshold=0.7, max_keep=10)

    assert [c.id for c in kept] == [1]
    assert next(s for s in scored if s.chunk.id == 2).relevance == 0.0


async def test_total_scoring_failure_yields_empty_keep_set() -> None:
    """If every score fails we keep nothing, so the caller refuses. Fail-closed."""
    items = [_chunk(1, "a"), _chunk(2, "b")]
    jev = _FixedScoreJev({}, fail_on={"a", "b"})

    _, kept = await triage_chunks(jev, "q", items, threshold=0.7, max_keep=10)

    assert kept == []


async def test_scoring_is_concurrent_not_sequential() -> None:
    """Regression guard against reintroducing a serial `for` loop.

    Ten chunks at 50ms each take ~500ms serially and ~50ms concurrently.
    """
    items = [_chunk(i, f"chunk {i}") for i in range(10)]
    jev = FakeJevClient(latency_s=0.05)

    start = time.perf_counter()
    await triage_chunks(jev, "q", items, threshold=0.0, max_keep=10, max_concurrency=10)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.25, f"triage appears serialised ({elapsed:.3f}s for 10x50ms)"
    assert jev.calls["score"] == 10


async def test_concurrency_limit_is_respected() -> None:
    """Unbounded fan-out would just move the bottleneck to the Jev service."""
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    class _Tracking(_FixedScoreJev):
        async def score(self, question: str, document: str) -> float:
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            return 0.9

    items = [_chunk(i, f"c{i}") for i in range(20)]
    await triage_chunks(_Tracking({}), "q", items, threshold=0.7, max_keep=20, max_concurrency=4)

    assert peak <= 4, f"concurrency limit exceeded: peak={peak}"


async def test_every_candidate_appears_in_the_audit_trail() -> None:
    """The trace must account for rejected chunks too, or triage is unauditable."""
    items = [_chunk(i, f"c{i}") for i in range(1, 6)]
    jev = _FixedScoreJev({f"c{i}": i / 10 for i in range(1, 6)})

    scored, _ = await triage_chunks(jev, "q", items, threshold=0.4, max_keep=10)

    assert {s.chunk.id for s in scored} == {1, 2, 3, 4, 5}
