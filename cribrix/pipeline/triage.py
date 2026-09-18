"""Stage 3 — Context triage (the cribrix).

Scores every retrieved chunk against the question and discards those below the
relevance threshold. This is the stage the whole project is named for.

Two implementation details matter more than they look:

**Concurrency.** Scoring N chunks is embarrassingly parallel. A `for` loop
turns a 20ms model into a 400ms stage at top_k=20. We fan out with
``asyncio.gather`` under a semaphore — unbounded fan-out would just move the
bottleneck to the Jev service and trigger rate limiting.

**Per-chunk failure isolation.** One failed score must not sink the request.
Failed chunks are scored 0.0 and dropped; if *every* score fails the caller
sees an empty result and refuses, which is the correct fail-closed behaviour.
"""

from __future__ import annotations

import asyncio

from cribrix.clients.jev import JevClient, JevError
from cribrix.observability import get_logger
from cribrix.schemas import Chunk, ScoredChunk

logger = get_logger(__name__)


async def _score_one(
    jev: JevClient, semaphore: asyncio.Semaphore, query: str, chunk: Chunk
) -> float:
    """Score a single chunk, degrading to 0.0 on failure.

    Per-chunk isolation matters: one failed scoring call must not sink the
    whole request. A failure scores 0.0, which drops that chunk. If *every*
    call fails the caller sees an empty keep-set and refuses — fail-closed.
    """
    async with semaphore:
        try:
            return await jev.score(question=query, document=chunk.content)
        except JevError:
            logger.warning("triage.score_failed", chunk_id=chunk.id, exc_info=True)
            return 0.0


async def triage_chunks(
    jev: JevClient,
    query: str,
    chunks: list[Chunk],
    *,
    threshold: float,
    max_keep: int,
    max_concurrency: int = 16,
) -> tuple[list[ScoredChunk], list[Chunk]]:
    """Score and filter retrieved chunks.

    Args:
        jev: System-1 client.
        query: The user's original question — the relevance reference.
        chunks: Candidates from retrieval.
        threshold: Minimum relevance to survive (inclusive).
        max_keep: Cap on survivors forwarded to the generator.
        max_concurrency: Bound on simultaneous in-flight score calls.

    Returns:
        ``(all_scored, kept)`` where ``all_scored`` is every chunk with its
        score and keep-flag (for the audit trace), and ``kept`` is the
        surviving chunks ordered most-relevant-first, truncated to `max_keep`.
    """
    if not chunks:
        return [], []

    semaphore = asyncio.Semaphore(max_concurrency)
    scores = await asyncio.gather(*(_score_one(jev, semaphore, query, chunk) for chunk in chunks))

    ranked = sorted(zip(chunks, scores, strict=True), key=lambda p: p[1], reverse=True)

    kept: list[Chunk] = []
    all_scored: list[ScoredChunk] = []
    for chunk, score in ranked:
        # Keep only if it clears the bar AND we still have budget.
        passes = score >= threshold and len(kept) < max_keep
        all_scored.append(ScoredChunk(chunk=chunk, relevance=score, kept=passes))
        if passes:
            kept.append(chunk)

    logger.info(
        "triage.completed",
        candidates=len(chunks),
        kept=len(kept),
        threshold=threshold,
        top_score=round(ranked[0][1], 4) if ranked else None,
    )
    return all_scored, kept
