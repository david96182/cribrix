"""Tests for the evaluation harness itself.

A harness that scores wrongly is worse than none: it produces confident,
publishable numbers that mean nothing. These pin down the scoring rules.
"""

from __future__ import annotations

import pytest

from cribrix.clients.jev import FakeJevClient
from cribrix.clients.llm import FakeLLMClient
from cribrix.config import Settings
from cribrix.evaluation.dataset import CORPUS, GOLDEN_CASES, GoldenCase, facts_present
from cribrix.evaluation.runner import (
    DEFAULT_RECORDING,
    Rate,
    cribrix_settings,
    evaluate,
    score_case,
    sweep,
)
from cribrix.pipeline.retrieval import HashingEmbedder, InMemoryVectorRetriever
from cribrix.schemas import AnswerStatus

# --- dataset ---------------------------------------------------------------


def test_golden_set_is_large_and_balanced() -> None:
    kinds = [c.kind for c in GOLDEN_CASES]
    assert len(GOLDEN_CASES) >= 50
    assert kinds.count("answerable") >= 25
    assert kinds.count("unanswerable") >= 15
    assert kinds.count("chitchat") >= 5


def test_every_answerable_case_has_facts_and_ground_truth_in_the_corpus() -> None:
    corpus = {c.id: c.content for c in CORPUS}
    for case in GOLDEN_CASES:
        if case.kind != "answerable":
            continue
        assert case.required_facts, case.query
        assert case.relevant_ids <= corpus.keys(), case.query
        evidence = " ".join(corpus[i] for i in case.relevant_ids)
        assert facts_present(evidence, case.required_facts), (
            f"label error: facts for {case.query!r} are not in its relevant chunks"
        )


def test_near_misses_are_labelled() -> None:
    assert sum(c.flavour == "near_miss" for c in GOLDEN_CASES) >= 8


def test_queries_are_unique() -> None:
    queries = [c.query for c in GOLDEN_CASES]
    assert len(queries) == len(set(queries))


# --- fact matching ---------------------------------------------------------


def test_facts_accept_alternatives_and_ignore_case() -> None:
    assert facts_present("The limit is 1,000 per minute.", ("1000|1,000",))
    assert facts_present("Uses AES-256.", ("aes-256",))
    assert not facts_present("The limit is 500.", ("1000|1,000",))


def test_facts_normalise_typographic_variants() -> None:
    """LLMs emit non-breaking hyphens and narrow spaces; they must not cause misses."""
    assert facts_present("encrypted using AES\u2011256", ("AES-256",))
    assert facts_present("within\u202f30\u202fdays", ("30 days",))


def test_all_facts_are_required() -> None:
    assert not facts_present("Support is in German.", ("German", "Japanese"))


# --- scoring rules ---------------------------------------------------------

ANSWERABLE = GoldenCase("q", "answerable", frozenset({1}), ("30 days",))
UNANSWERABLE = GoldenCase("q", "unanswerable")
CHITCHAT = GoldenCase("q", "chitchat")


def test_answered_with_correct_facts_is_correct() -> None:
    assert score_case(ANSWERABLE, AnswerStatus.ANSWERED, "It is 30 days.") == (True, False)


def test_answered_with_wrong_facts_is_a_hallucination() -> None:
    """'Faithfully grounded in the wrong passage' must be scored as a failure."""
    assert score_case(ANSWERABLE, AnswerStatus.ANSWERED, "It is 14 days.") == (False, True)


def test_refusing_an_answerable_question_is_a_miss_not_a_hallucination() -> None:
    assert score_case(ANSWERABLE, AnswerStatus.INSUFFICIENT_CONTEXT, "") == (False, False)


@pytest.mark.parametrize(
    "status",
    [AnswerStatus.INSUFFICIENT_CONTEXT, AnswerStatus.DECLINED, AnswerStatus.UNGROUNDED],
)
def test_any_refusal_of_an_unanswerable_question_is_correct(status: AnswerStatus) -> None:
    assert score_case(UNANSWERABLE, status, "") == (True, False)


def test_answering_an_unanswerable_question_is_a_hallucination() -> None:
    assert score_case(UNANSWERABLE, AnswerStatus.ANSWERED, "10%") == (False, True)


def test_chitchat_scoring() -> None:
    assert score_case(CHITCHAT, AnswerStatus.CHITCHAT, "hi")[0] is True
    assert score_case(CHITCHAT, AnswerStatus.ANSWERED, "policy")[0] is False


# --- baseline refusal crediting ---------------------------------------------


class _Says(FakeLLMClient):
    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    async def generate(self, query, chunks, *, system_prompt=None):  # type: ignore[no-untyped-def]
        return self._text


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("**I don't know.**\n\nThe passages only cover laptops for marketing.", "DECLINED"),
        ("There is no SLA guarantee for Pro plan customers.", "DECLINED"),
        ("**30 days**\n\nEnterprise customers (which includes Business)...", "ANSWERED"),
        ("**No**, the company does not offer a pension scheme.", "ANSWERED"),
    ],
)
async def test_naive_baseline_is_credited_for_leading_refusals(text: str, status: str) -> None:
    """The baseline gets the benefit of the doubt, but not for asserted answers."""
    from cribrix.evaluation.runner import run_naive

    retriever = InMemoryVectorRetriever(CORPUS, HashingEmbedder(dimension=384))
    got, _, _ = await run_naive(_Says(text), retriever, UNANSWERABLE)
    assert got.value == status


# --- statistics ------------------------------------------------------------


def test_wilson_interval_is_sane() -> None:
    lo, hi = Rate(0, 22).interval()
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert 0.1 < hi < 0.2  # zero hits still has a non-zero upper bound
    lo, hi = Rate(10, 20).interval()
    assert lo < 0.5 < hi
    assert Rate(0, 0).interval() == (0.0, 0.0)


# --- harness ---------------------------------------------------------------


def test_evaluation_pins_pipeline_thresholds_to_code_defaults() -> None:
    """A local .env must not change the published numbers."""
    tweaked = Settings(relevance_threshold=0.99, evidence_threshold=0.0)
    pinned = cribrix_settings(tweaked)
    assert pinned.relevance_threshold == Settings.model_fields["relevance_threshold"].default
    assert pinned.evidence_threshold == Settings.model_fields["evidence_threshold"].default
    assert cribrix_settings(tweaked, relevance_threshold=0.3).relevance_threshold == 0.3


async def test_vector_retriever_ranks_like_cosine_search() -> None:
    retriever = InMemoryVectorRetriever(CORPUS, HashingEmbedder(dimension=384))
    results = await retriever.retrieve("engineering team laptop MacBook", top_k=3)
    assert results[0].content.startswith("The engineering team uses")
    assert results[0].distance <= results[1].distance <= results[2].distance


async def test_fake_backend_runs_end_to_end() -> None:
    settings = cribrix_settings(Settings(), retrieval_top_k=20, embedding_dim=384)
    naive, guarded, responses = await evaluate(FakeJevClient(), FakeLLMClient(), settings)
    assert len(naive.cases) == len(guarded.cases) == len(responses) == len(GOLDEN_CASES)
    # The naive system has no routing, so it can never handle chitchat.
    assert naive.chitchat_accuracy.hits == 0
    assert guarded.unanswerable_answered.hits < naive.unanswerable_answered.hits


async def test_sweep_rescoring_needs_no_api_calls() -> None:
    jev = FakeJevClient()
    settings = cribrix_settings(Settings(), retrieval_top_k=20, embedding_dim=384)
    _, _, responses = await evaluate(jev, FakeLLMClient(), settings)
    before = dict(jev.calls)
    rows = sweep(responses, GOLDEN_CASES, [0.0, 0.5, 1.01], [0.0], injection_max=1.0)
    assert jev.calls == before
    # A threshold above 1.0 keeps nothing: every unanswerable case "refuses".
    last = rows[-1]
    assert last[2] == 0 and last[4] == last[5]


def test_committed_recording_exists() -> None:
    """CI replays this file; it must ship with the repo."""
    assert DEFAULT_RECORDING.exists()
