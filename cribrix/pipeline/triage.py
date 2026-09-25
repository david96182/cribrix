"""Stage 3 — Context triage (the cribrix).

Each retrieved chunk gets **one** Jev request carrying three independent
questions about the (question, passage) pair:

``relevance``  Score over a 4-level rubric, normalised to 0..1.
``answers``    Noul: does the passage actually *state* what was asked?
``injection``  Noul: is the passage trying to instruct the model?

The decision is made in code, first match wins, so every threshold is a
reviewable constant rather than a reworded prompt:

1. ``injection > injection_max``       -> drop (security, checked first)
2. ``relevance < relevance_threshold`` -> drop (off topic)
3. ``answers < evidence_threshold``    -> drop (on topic, but not an answer)
4. otherwise                           -> keep

Rule 3 is what separates *relevance* from *answerability*: a refund-window
passage is topically close to "what is the early termination penalty?", but it
does not state a penalty, so it should not reach the generator.

Because the three questions share one request, the extra signals cost almost
no latency (questions are evaluated in parallel against one state).

**Concurrency.** Chunks are assessed concurrently under a semaphore; unbounded
fan-out would just move the bottleneck to the Jev rate limiter.

**Per-chunk failure isolation.** A failed request drops only that chunk. If
every request fails the keep-set is empty and the caller refuses — fail-closed.
"""

from __future__ import annotations

import asyncio

from cribrix.clients.jev import JevClient, JevError, PassageAssessment
from cribrix.observability import get_logger
from cribrix.schemas import Chunk, ScoredChunk

logger = get_logger(__name__)


async def _assess_one(
    jev: JevClient, semaphore: asyncio.Semaphore, query: str, chunk: Chunk
) -> PassageAssessment | None:
    """Assess a single chunk; ``None`` means the call failed."""
    async with semaphore:
        try:
            return await jev.assess_passage(question=query, passage=chunk.content)
        except JevError:
            logger.warning("triage.assess_failed", chunk_id=chunk.id, exc_info=True)
            return None


def _decide(
    assessment: PassageAssessment | None,
    *,
    threshold: float,
    evidence_threshold: float,
    injection_max: float,
) -> str | None:
    """Return the drop reason, or ``None`` to keep. First match wins."""
    if assessment is None:
        return "assessment_failed"
    if assessment.injection > injection_max:
        return "prompt_injection"
    if assessment.relevance < threshold:
        return "irrelevant"
    if assessment.answers < evidence_threshold:
        return "does_not_answer"
    return None


async def triage_chunks(
    jev: JevClient,
    query: str,
    chunks: list[Chunk],
    *,
    threshold: float,
    max_keep: int,
    evidence_threshold: float = 0.0,
    injection_max: float = 1.0,
    max_concurrency: int = 16,
) -> tuple[list[ScoredChunk], list[Chunk]]:
    """Assess and filter retrieved chunks.

    Args:
        jev: System-1 client.
        query: The user's original question — the relevance reference.
        chunks: Candidates from retrieval.
        threshold: Minimum normalised relevance to survive (inclusive).
        max_keep: Cap on survivors forwarded to the generator.
        evidence_threshold: Minimum ``answers`` probability (inclusive).
            0.0 disables the answerability check.
        injection_max: Chunks with an ``injection`` probability above this are
            dropped. 1.0 disables the check.
        max_concurrency: Bound on simultaneous in-flight requests.

    Returns:
        ``(all_scored, kept)`` where ``all_scored`` is every chunk with its
        signals and verdict (for the audit trace), and ``kept`` is the
        survivors ordered most-relevant-first, truncated to `max_keep`.
    """
    if not chunks:
        return [], []

    semaphore = asyncio.Semaphore(max_concurrency)
    assessments = await asyncio.gather(
        *(_assess_one(jev, semaphore, query, chunk) for chunk in chunks)
    )

    def sort_key(pair: tuple[Chunk, PassageAssessment | None]) -> tuple[float, float]:
        a = pair[1]
        return (a.relevance, a.answers) if a else (-1.0, -1.0)

    ranked = sorted(zip(chunks, assessments, strict=True), key=sort_key, reverse=True)

    kept: list[Chunk] = []
    all_scored: list[ScoredChunk] = []
    for chunk, assessment in ranked:
        reason = _decide(
            assessment,
            threshold=threshold,
            evidence_threshold=evidence_threshold,
            injection_max=injection_max,
        )
        if reason is None and len(kept) >= max_keep:
            reason = "over_budget"
        passes = reason is None
        all_scored.append(
            ScoredChunk(
                chunk=chunk,
                relevance=assessment.relevance if assessment else 0.0,
                answers=assessment.answers if assessment else 0.0,
                injection=assessment.injection if assessment else 0.0,
                kept=passes,
                drop_reason=reason,
            )
        )
        if passes:
            kept.append(chunk)

    logger.info(
        "triage.completed",
        candidates=len(chunks),
        kept=len(kept),
        threshold=threshold,
        evidence_threshold=evidence_threshold,
        top_score=round(all_scored[0].relevance, 4) if all_scored else None,
    )
    return all_scored, kept
