"""Pipeline orchestration: the full request flow, stage by stage.

    route -> retrieve -> triage -> generate -> verify

Every stage can short-circuit to a *typed refusal*. That is the central design
commitment: the system distinguishes "the corpus is empty", "nothing was
relevant", and "the draft was ungrounded", because those are three different
bugs with three different fixes. Collapsing them into one generic error string
is what makes most RAG systems undebuggable.

The orchestrator depends only on Protocols (`Retriever`, `JevClient`,
`LLMClient`), so the entire flow is unit-testable without Postgres or network.
"""

from __future__ import annotations

import time

from cribrix.clients.jev import JevClient
from cribrix.clients.llm import LLMClient, LLMError
from cribrix.config import Settings
from cribrix.observability import get_logger, new_request_id, set_request_id, stage_timer
from cribrix.pipeline.retrieval import Retriever
from cribrix.pipeline.router import route_intent
from cribrix.pipeline.triage import triage_chunks
from cribrix.pipeline.verification import verify_answer
from cribrix.schemas import (
    REFUSAL_MESSAGE,
    AnswerStatus,
    Chunk,
    Intent,
    PipelineTrace,
    QueryResponse,
    ScoredChunk,
    SourceRef,
)

logger = get_logger(__name__)

CHITCHAT_REPLY = (
    "Hello. I'm a document question-answering assistant — ask me something about "
    "the indexed corpus and I'll answer only from what the sources actually support."
)


def _sources_from(scored: list[ScoredChunk], kept: list[Chunk]) -> list[SourceRef]:
    """Build citations for the chunks that actually fed the answer."""
    relevance = {s.chunk.id: s.relevance for s in scored}
    return [
        SourceRef(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            relevance=round(relevance.get(chunk.id, 0.0), 4),
            excerpt=chunk.content[:280] + ("..." if len(chunk.content) > 280 else ""),
        )
        for chunk in kept
    ]


class RAGPipeline:
    """Coordinates the five pipeline stages for a single query."""

    def __init__(
        self,
        *,
        jev: JevClient,
        llm: LLMClient,
        retriever: Retriever,
        settings: Settings,
    ) -> None:
        self._jev = jev
        self._llm = llm
        self._retriever = retriever
        self._settings = settings

    async def run(self, query: str, *, include_trace: bool = True) -> QueryResponse:
        """Execute the pipeline end to end.

        Args:
            query: The user's question.
            include_trace: Attach the full per-stage audit trail to the response.

        Returns:
            A `QueryResponse` whose `status` names the exact terminal state.
        """
        request_id = new_request_id()
        set_request_id(request_id)
        trace = PipelineTrace(request_id=request_id)
        started = time.perf_counter()

        try:
            return await self._run_stages(query, trace, include_trace)
        finally:
            trace.total_ms = round((time.perf_counter() - started) * 1000.0, 3)
            logger.info(
                "pipeline.finished",
                total_ms=trace.total_ms,
                stages={t.stage: t.duration_ms for t in trace.timings},
            )

    # -- internals ----------------------------------------------------------

    async def _run_stages(
        self, query: str, trace: PipelineTrace, include_trace: bool
    ) -> QueryResponse:
        cfg = self._settings

        def finish(
            status: AnswerStatus,
            answer: str,
            *,
            sources: list[SourceRef] | None = None,
            verified: bool = False,
        ) -> QueryResponse:
            return QueryResponse(
                status=status,
                answer=answer,
                sources=sources or [],
                verified=verified,
                trace=trace if include_trace else None,
            )

        # -- Stage 1: routing ----------------------------------------------
        with stage_timer(trace, "routing"):
            intent, confidence = await route_intent(
                self._jev, query, chitchat_min_confidence=cfg.chitchat_min_confidence
            )
        trace.jev_requests += 1
        trace.intent = intent
        trace.intent_confidence = round(confidence, 4)

        if intent is Intent.CHITCHAT:
            trace.notes.append("routed to chitchat; retrieval skipped")
            return finish(AnswerStatus.CHITCHAT, CHITCHAT_REPLY, verified=True)

        # -- Stage 2: retrieval ---------------------------------------------
        with stage_timer(trace, "retrieval"):
            candidates = await self._retriever.retrieve(query, top_k=cfg.retrieval_top_k)
        trace.retrieved_count = len(candidates)

        if not candidates:
            # Distinct from INSUFFICIENT_CONTEXT: the corpus itself is empty.
            trace.notes.append("retrieval returned zero chunks")
            return finish(AnswerStatus.NO_DOCUMENTS, REFUSAL_MESSAGE)

        # -- Stage 3: triage (the cribrix) ------------------------------------
        with stage_timer(trace, "triage"):
            scored, kept = await triage_chunks(
                self._jev,
                query,
                candidates,
                threshold=cfg.relevance_threshold,
                evidence_threshold=cfg.evidence_threshold,
                injection_max=cfg.injection_max,
                max_keep=cfg.max_chunks_to_llm,
                max_concurrency=cfg.jev_max_concurrency,
            )
        trace.jev_requests += len(candidates)
        trace.scored_chunks = scored
        trace.kept_count = len(kept)

        # The critical guard. Calling the LLM with thin or empty context is the
        # single most reliable way to manufacture a hallucination, so we refuse
        # *before* spending a generation call.
        if len(kept) < cfg.min_chunks_required or not kept:
            trace.notes.append(
                f"only {len(kept)} chunk(s) survived triage "
                f"(relevance>={cfg.relevance_threshold}, evidence>={cfg.evidence_threshold}); "
                "refusing without generating"
            )
            return finish(AnswerStatus.INSUFFICIENT_CONTEXT, REFUSAL_MESSAGE)

        # -- Stage 4: generation --------------------------------------------
        with stage_timer(trace, "generation"):
            trace.llm_calls += 1
            try:
                draft = await self._llm.generate(query, kept)
            except LLMError:
                logger.error("generation.failed", exc_info=True)
                trace.notes.append("generator error")
                # An outage is not a verdict on the evidence; report it as such.
                return finish(AnswerStatus.GENERATION_FAILED, REFUSAL_MESSAGE)

        # -- Stage 5: verification ------------------------------------------
        with stage_timer(trace, "verification"):
            result = await verify_answer(
                self._jev,
                kept,
                draft,
                question=query,
                mode=cfg.verification_mode,
                groundedness_threshold=cfg.groundedness_threshold,
                noul_threshold=cfg.noul_threshold,
                fail_open=cfg.fail_open_on_verifier_error,
            )
        if result.verdicts or result.reason == "verifier_unavailable":
            trace.jev_requests += 1
        trace.claim_verdicts = result.verdicts
        trace.groundedness = result.groundedness
        trace.notes.append(f"verification: {result.reason}")

        if result.declined:
            # The generator read the relevant evidence and honestly said it does
            # not contain the answer. That is a correct refusal, not an answer.
            return finish(AnswerStatus.DECLINED, draft)

        if not result.passed:
            if result.reason == "verifier_unavailable":
                return finish(AnswerStatus.VERIFIER_UNAVAILABLE, REFUSAL_MESSAGE)
            # The draft existed but wasn't supported — withhold it entirely.
            return finish(AnswerStatus.UNGROUNDED, REFUSAL_MESSAGE)

        verified = result.reason != "verifier_unavailable_failed_open"
        if not verified:
            trace.notes.append("verifier unavailable; answer released unverified (fail-open)")
        return finish(
            AnswerStatus.ANSWERED,
            draft,
            sources=_sources_from(scored, kept),
            verified=verified,
        )
