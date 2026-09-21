"""End-to-end pipeline tests, exercising stage interaction and short-circuits.

These assert on `AnswerStatus` rather than on answer text. The status *is* the
contract: it tells the caller exactly which stage terminated the request.
"""

from __future__ import annotations

from cribrix.clients.jev import FakeJevClient
from cribrix.clients.llm import FakeLLMClient
from cribrix.config import Settings
from cribrix.pipeline.orchestrator import RAGPipeline
from cribrix.pipeline.retrieval import InMemoryRetriever
from cribrix.schemas import REFUSAL_MESSAGE, AnswerStatus, Chunk, Intent


def _pipeline(
    settings: Settings,
    chunks: list[Chunk],
    *,
    jev: FakeJevClient | None = None,
    llm: FakeLLMClient | None = None,
) -> RAGPipeline:
    return RAGPipeline(
        jev=jev or FakeJevClient(),
        llm=llm or FakeLLMClient(),
        retriever=InMemoryRetriever(chunks),
        settings=settings,
    )


# --- happy path ------------------------------------------------------------


async def test_relevant_question_is_answered_and_verified(
    settings: Settings, chunks: list[Chunk]
) -> None:
    llm = FakeLLMClient()
    response = await _pipeline(settings, chunks, llm=llm).run(
        "What is the refund window for enterprise customers?"
    )

    assert response.status is AnswerStatus.ANSWERED
    assert response.verified is True
    assert response.sources
    assert llm.call_count == 1


async def test_answer_cites_only_chunks_that_survived_triage(
    settings: Settings, chunks: list[Chunk]
) -> None:
    """Citations must reflect the evidence actually used, not the raw retrieval."""
    response = await _pipeline(settings, chunks).run(
        "What is the refund window for enterprise customers?"
    )

    assert response.trace is not None
    kept_ids = {s.chunk.id for s in response.trace.scored_chunks if s.kept}
    assert {s.chunk_id for s in response.sources} == kept_ids
    assert all(s.relevance >= settings.relevance_threshold for s in response.sources)


# --- routing short-circuit -------------------------------------------------


async def test_chitchat_skips_retrieval_entirely(settings: Settings, chunks: list[Chunk]) -> None:
    llm = FakeLLMClient()
    response = await _pipeline(settings, chunks, llm=llm).run("hello there!")

    assert response.status is AnswerStatus.CHITCHAT
    assert response.trace is not None
    assert response.trace.intent is Intent.CHITCHAT
    assert response.trace.retrieved_count == 0
    assert llm.call_count == 0, "chitchat must not spend a generation call"


# --- refusal paths ---------------------------------------------------------


async def test_empty_corpus_reports_no_documents(settings: Settings) -> None:
    """Distinct from INSUFFICIENT_CONTEXT: this means the index is empty."""
    response = await _pipeline(settings, []).run("What is the refund window?")

    assert response.status is AnswerStatus.NO_DOCUMENTS
    assert response.answer == REFUSAL_MESSAGE
    assert response.verified is False


async def test_irrelevant_corpus_refuses_without_calling_the_llm(
    settings: Settings, chunks: list[Chunk]
) -> None:
    """The core guard.

    A question with no supporting evidence must be refused *before* generation.
    Calling the LLM with thin context is how RAG systems fabricate answers, so
    the assertion that matters here is `llm.call_count == 0`.
    """
    strict = settings.model_copy(update={"relevance_threshold": 0.95})
    llm = FakeLLMClient()

    response = await _pipeline(strict, chunks, llm=llm).run(
        "What was the company's total revenue in fiscal year 2019?"
    )

    assert response.status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert response.answer == REFUSAL_MESSAGE
    assert llm.call_count == 0, "generation must not run on empty context"


async def test_hallucinated_answer_is_blocked(settings: Settings, chunks: list[Chunk]) -> None:
    """The gate must withhold a draft containing a fabricated claim."""
    liar = FakeLLMClient(hallucinate=True)

    response = await _pipeline(settings, chunks, llm=liar).run(
        "What is the refund window for enterprise customers?"
    )

    assert liar.call_count == 1, "the draft was generated..."
    assert response.status is AnswerStatus.UNGROUNDED, "...and then withheld"
    assert response.answer == REFUSAL_MESSAGE
    assert "Zanzibar" not in response.answer


async def test_generator_failure_degrades_to_refusal(
    settings: Settings, chunks: list[Chunk]
) -> None:
    response = await _pipeline(settings, chunks, llm=FakeLLMClient(fail=True)).run(
        "What is the refund window for enterprise customers?"
    )
    assert response.status is AnswerStatus.UNGROUNDED
    assert response.verified is False


async def test_verifier_outage_fails_closed(settings: Settings, chunks: list[Chunk]) -> None:
    """With Jev down, routing defaults to SEARCH and triage keeps nothing.

    Every stage degrades safely; the request terminates in a refusal rather
    than an unverified answer.
    """
    response = await _pipeline(settings, chunks, jev=FakeJevClient(fail=True)).run(
        "What is the refund window?"
    )

    assert response.verified is False
    assert response.answer == REFUSAL_MESSAGE
    assert response.status in {
        AnswerStatus.INSUFFICIENT_CONTEXT,
        AnswerStatus.UNGROUNDED,
        AnswerStatus.VERIFIER_UNAVAILABLE,
    }


# --- configuration behaviour ----------------------------------------------


async def test_threshold_controls_how_much_context_survives(
    settings: Settings, chunks: list[Chunk]
) -> None:
    """Raising the threshold must monotonically reduce surviving context."""
    query = "What is the refund window for enterprise customers?"

    lenient = await _pipeline(settings.model_copy(update={"relevance_threshold": 0.0}), chunks).run(
        query
    )
    strict = await _pipeline(settings.model_copy(update={"relevance_threshold": 0.9}), chunks).run(
        query
    )

    assert lenient.trace is not None and strict.trace is not None
    assert lenient.trace.kept_count >= strict.trace.kept_count


async def test_max_chunks_to_llm_caps_forwarded_context(
    settings: Settings, chunks: list[Chunk]
) -> None:
    capped = settings.model_copy(update={"relevance_threshold": 0.0, "max_chunks_to_llm": 2})
    response = await _pipeline(capped, chunks).run("refund window enterprise")

    assert response.trace is not None
    assert response.trace.kept_count <= 2


# --- trace and observability ----------------------------------------------


async def test_trace_records_every_stage(settings: Settings, chunks: list[Chunk]) -> None:
    response = await _pipeline(settings, chunks).run(
        "What is the refund window for enterprise customers?"
    )

    assert response.trace is not None
    stages = {t.stage for t in response.trace.timings}
    assert {"routing", "retrieval", "triage", "generation", "verification"} <= stages
    assert response.trace.total_ms > 0
    assert all(t.duration_ms >= 0 for t in response.trace.timings)


async def test_trace_can_be_suppressed(settings: Settings, chunks: list[Chunk]) -> None:
    response = await _pipeline(settings, chunks).run("refund window?", include_trace=False)
    assert response.trace is None


async def test_each_request_gets_a_unique_id(settings: Settings, chunks: list[Chunk]) -> None:
    pipe = _pipeline(settings, chunks)
    first = await pipe.run("refund window enterprise?")
    second = await pipe.run("refund window enterprise?")

    assert first.trace is not None and second.trace is not None
    assert first.trace.request_id != second.trace.request_id
