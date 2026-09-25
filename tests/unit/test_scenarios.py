"""Tests for the three demonstration scenarios.

These assert the *architectural* claims each scenario makes, offline and
deterministically. The live run proves the same things against real APIs, but
CI needs a version that cannot flake.
"""

from __future__ import annotations

import pytest

from cribrix.clients.jev import FakeJevClient
from cribrix.clients.llm import FakeLLMClient
from cribrix.config import Settings
from cribrix.evaluation.scenarios import (
    SCENARIO_CHITCHAT,
    SCENARIO_HALLUCINATION,
    SCENARIO_MIRAGE,
    SCENARIOS,
    probe_verifier,
    run_cribrix,
    run_naive,
)
from cribrix.schemas import AnswerStatus


@pytest.fixture
def settings() -> Settings:
    # 0.6 is tuned to the *fake's* token-coverage heuristic, where the
    # same-vocabulary distractor lands exactly on 0.5. These tests check the
    # wiring of each scenario, not the calibration of the real model.
    return Settings(
        relevance_threshold=0.6,
        evidence_threshold=0.5,
        min_chunks_required=1,
        retrieval_top_k=10,
        max_chunks_to_llm=5,
        verification_mode="atomic",
        groundedness_threshold=1.0,
        noul_threshold=0.5,
        jev_mode="fake",
        llm_provider="fake",
        env="test",
        log_json=False,
    )


# --- Scenario 1: the chitchat trap -----------------------------------------


async def test_chitchat_skips_retrieval_and_the_llm(settings: Settings) -> None:
    """The saving is not just tokens: the vector search never runs either."""
    llm = FakeLLMClient()
    result, response = await run_cribrix(SCENARIO_CHITCHAT, FakeJevClient(), llm, settings)

    assert response.status is AnswerStatus.CHITCHAT
    assert result.searched is False
    assert result.llm_called is False
    assert result.prompt_tokens == 0
    assert llm.call_count == 0


async def test_naive_pipeline_wastes_a_search_and_a_generation(settings: Settings) -> None:
    """The baseline retrieves HR policy documents in reply to a greeting."""
    llm = FakeLLMClient()
    naive = await run_naive(SCENARIO_CHITCHAT, llm)

    assert naive.searched is True
    assert naive.llm_called is True
    assert naive.chunks_to_llm == 3
    assert naive.prompt_tokens > 0


async def test_chitchat_saves_one_hundred_percent_of_prompt_tokens(
    settings: Settings,
) -> None:
    naive = await run_naive(SCENARIO_CHITCHAT, FakeLLMClient())
    guarded, _ = await run_cribrix(SCENARIO_CHITCHAT, FakeJevClient(), FakeLLMClient(), settings)
    assert naive.prompt_tokens > 0
    assert guarded.prompt_tokens == 0


# --- Scenario 2: the keyword mirage ----------------------------------------


async def test_triage_drops_both_distractors(settings: Settings) -> None:
    """'MacBook Airs' and 'mac and cheese' are lexical neighbours, not answers."""
    _, response = await run_cribrix(SCENARIO_MIRAGE, FakeJevClient(), FakeLLMClient(), settings)

    assert response.status is AnswerStatus.ANSWERED
    assert response.trace is not None
    kept = [s for s in response.trace.scored_chunks if s.kept]
    assert len(kept) == 1
    assert "engineering" in kept[0].chunk.content.lower()

    dropped = {s.chunk.content for s in response.trace.scored_chunks if not s.kept}
    assert any("marketing" in d.lower() for d in dropped)
    assert any("cheese" in d.lower() for d in dropped)


async def test_relevance_ordering_is_correct(settings: Settings) -> None:
    """The distractor must outrank the noise, and both must lose to the answer."""
    _, response = await run_cribrix(SCENARIO_MIRAGE, FakeJevClient(), FakeLLMClient(), settings)
    assert response.trace is not None
    by_content = {s.chunk.content: s.relevance for s in response.trace.scored_chunks}
    answer = next(v for k, v in by_content.items() if "engineering" in k.lower())
    distractor = next(v for k, v in by_content.items() if "marketing" in k.lower())
    noise = next(v for k, v in by_content.items() if "cheese" in k.lower())
    assert answer > distractor >= noise


async def test_mirage_reduces_prompt_size(settings: Settings) -> None:
    naive = await run_naive(SCENARIO_MIRAGE, FakeLLMClient())
    guarded, _ = await run_cribrix(SCENARIO_MIRAGE, FakeJevClient(), FakeLLMClient(), settings)
    assert guarded.chunks_to_llm < naive.chunks_to_llm
    assert guarded.prompt_tokens < naive.prompt_tokens


# --- Scenario 3: the confident hallucination -------------------------------


async def test_fabricated_percentage_is_blocked(settings: Settings) -> None:
    """A generator that invents '10 percent' must not reach the user.

    The evidence check is relaxed here (as the offline scenario does) so the
    fabricated draft actually reaches the gate under test.
    """
    liar = FakeLLMClient(hallucinate=True)
    relaxed = settings.model_copy(update={"evidence_threshold": 0.0})
    result, response = await run_cribrix(SCENARIO_HALLUCINATION, FakeJevClient(), liar, relaxed)

    assert liar.call_count == 1, "the draft was generated..."
    assert response.status is AnswerStatus.UNGROUNDED, "...and then withheld"
    assert "10 percent" not in response.answer
    assert result.status == AnswerStatus.UNGROUNDED.value


async def test_adversarial_probe_isolates_the_gate(settings: Settings) -> None:
    """The probe tests Cribrix, not the generator.

    Whether a given model hallucinates is a property of that model. Whether
    the gate catches a fabrication is a property of Cribrix, and that is what
    this measures.
    """
    probe = await probe_verifier(SCENARIO_HALLUCINATION, FakeJevClient(), settings)

    assert probe is not None
    blocked, verdicts = probe
    assert blocked is False, "the fabricated draft must not pass the gate"
    fabricated = next(v for v in verdicts if "10%" in v.claim)
    assert fabricated.grounded is False
    assert fabricated.unsupported_numbers == ["10"]


async def test_probe_returns_none_without_an_adversarial_draft(settings: Settings) -> None:
    assert await probe_verifier(SCENARIO_MIRAGE, FakeJevClient(), settings) is None


# --- Suite-level -----------------------------------------------------------


def test_every_scenario_is_distinct_and_documented() -> None:
    assert len({s.key for s in SCENARIOS}) == len(SCENARIOS) == 3
    for scenario in SCENARIOS:
        assert scenario.problem and scenario.naive_failure
        assert scenario.corpus, "a scenario needs a corpus to be meaningful"


def test_scenarios_cover_three_different_pipeline_stages() -> None:
    """Routing, triage and verification — one failure mode each."""
    assert {s.expected for s in SCENARIOS} == {
        AnswerStatus.CHITCHAT,
        AnswerStatus.ANSWERED,
        AnswerStatus.UNGROUNDED,
    }
