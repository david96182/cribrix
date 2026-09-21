"""Tests for the Jev and LLM client layers.

The fakes underpin every other test, so their determinism is load-bearing: if
they were nondeterministic, failures elsewhere would be unreproducible.

The `_normalise_score` tests are the most important in this file — they guard
the single most dangerous misreading of the TypeSafe SDK.
"""

from __future__ import annotations

import pytest

from cribrix.clients.jev import (
    RELEVANCE_RUBRIC,
    ROUTING_CRITERIA,
    FakeJevClient,
    JevClient,
    JevError,
    _normalise_score,
    build_jev_client,
)
from cribrix.clients.llm import (
    PROVIDER_DEFAULTS,
    AnthropicClient,
    FakeLLMClient,
    LLMClient,
    LLMError,
    OpenAICompatibleClient,
    _strip_reasoning,
    build_llm_client,
    build_prompt,
)
from cribrix.config import Settings
from cribrix.schemas import Chunk

# ---------------------------------------------------------------------------
# Score normalisation — the SDK's sharpest edge
# ---------------------------------------------------------------------------


def test_ordinal_score_is_normalised_to_unit_interval() -> None:
    """`Score` returns a rubric index, not a 0..1 float.

    A live 4-level rubric returns 3.0 for a direct answer. Comparing that raw
    value against a 0.7 threshold would keep every chunk and silently turn the
    triage stage into a no-op. This is the regression guard for that bug.
    """
    size = len(RELEVANCE_RUBRIC)
    assert _normalise_score(0.0, size) == 0.0
    assert _normalise_score(3.0, size) == 1.0
    assert _normalise_score(1.0, size) == pytest.approx(1 / 3)
    assert _normalise_score(2.0, size) == pytest.approx(2 / 3)


def test_normalised_score_is_clamped() -> None:
    """Out-of-range values from a future rubric change must not escape [0, 1]."""
    assert _normalise_score(99.0, 4) == 1.0
    assert _normalise_score(-5.0, 4) == 0.0


def test_degenerate_rubric_does_not_divide_by_zero() -> None:
    assert _normalise_score(1.0, 1) == 0.0


def test_raw_ordinal_would_defeat_the_threshold() -> None:
    """Documents the failure this normalisation prevents."""
    raw_direct_answer = 3.0
    raw_irrelevant = 1.0
    threshold = 0.7
    # Unnormalised, even an irrelevant chunk clears the bar.
    assert raw_irrelevant > threshold
    # Normalised, the two are correctly separated.
    assert _normalise_score(raw_irrelevant, 4) < threshold <= _normalise_score(raw_direct_answer, 4)


# ---------------------------------------------------------------------------
# Fake Jev
# ---------------------------------------------------------------------------


async def test_fake_jev_satisfies_the_protocol() -> None:
    assert isinstance(FakeJevClient(), JevClient)


async def test_choice_returns_a_label_from_the_criteria() -> None:
    jev = FakeJevClient()
    for text in ["hello there", "what is the refund policy", "12345"]:
        label, confidence = await jev.choice(context=text, options=ROUTING_CRITERIA)
        assert label in ROUTING_CRITERIA
        assert 0.0 <= confidence <= 1.0


async def test_choice_rejects_empty_options() -> None:
    with pytest.raises(JevError):
        await FakeJevClient().choice(context="x", options={})


async def test_score_is_bounded_and_deterministic() -> None:
    a = await FakeJevClient().score(question="refund window", document="refund window is 30 days")
    b = await FakeJevClient().score(question="refund window", document="refund window is 30 days")
    assert a == b
    assert 0.0 <= a <= 1.0


async def test_score_separates_relevant_from_irrelevant() -> None:
    jev = FakeJevClient()
    question = "What laptop does the engineering team use?"
    relevant = await jev.score(
        question=question, document="The engineering team uses MacBook Pro M3s."
    )
    noise = await jev.score(question=question, document="The cafeteria serves mac and cheese.")
    assert relevant > noise


async def test_score_batch_matches_sequential_scoring() -> None:
    jev = FakeJevClient()
    docs = ["alpha refund policy", "unrelated cafeteria text"]
    batch = await jev.score_batch("refund policy", docs)
    individual = [await jev.score("refund policy", d) for d in docs]
    assert batch == individual


async def test_score_batch_on_empty_input() -> None:
    assert await FakeJevClient().score_batch("q", []) == []


async def test_grounded_returns_a_probability_not_a_bool() -> None:
    """Noul is probabilistic; the pipeline owns the threshold decision."""
    value = await FakeJevClient().grounded(source="the sky is blue", claim="the sky is blue")
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


async def test_fabricated_number_scores_near_zero() -> None:
    """Scenario 3 in miniature: an invented figure must be detectable."""
    jev = FakeJevClient()
    source = "The company offers a bonus. The percentage is decided by the board in December."
    assert await jev.grounded(source=source, claim="The bonus is 10 percent.") < 0.1


async def test_supported_claim_scores_high() -> None:
    jev = FakeJevClient()
    source = "The company offers a bonus decided by the board every December."
    assert await jev.grounded(source=source, claim="The board decides the bonus.") > 0.5


async def test_nothing_is_grounded_in_an_empty_source() -> None:
    assert await FakeJevClient().grounded(source="", claim="some claim") == 0.0


async def test_fake_jev_failure_mode() -> None:
    jev = FakeJevClient(fail=True)
    with pytest.raises(JevError):
        await jev.score(question="a", document="b")
    assert await jev.health() is False


async def test_call_counts_are_tracked() -> None:
    jev = FakeJevClient()
    await jev.choice(context="x", options=ROUTING_CRITERIA)
    await jev.score(question="x", document="y")
    await jev.grounded(source="x", claim="y")
    assert jev.calls == {"choice": 1, "score": 1, "grounded": 1}


def test_jev_factory_requires_a_key_in_live_mode() -> None:
    with pytest.raises(JevError, match="API_KEY"):
        build_jev_client(Settings(jev_mode="live", jev_api_key=None))


def test_jev_factory_defaults_to_the_fake() -> None:
    assert isinstance(build_jev_client(Settings(jev_mode="fake")), FakeJevClient)


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------


async def test_fake_llm_satisfies_the_protocol() -> None:
    assert isinstance(FakeLLMClient(), LLMClient)


async def test_fake_llm_is_extractive_and_grounded_by_construction() -> None:
    """A mock that invented text would make the verification tests meaningless."""
    chunks = [Chunk(id=1, document_id="d", content="The refund window is 30 days.")]
    assert "30 days" in await FakeLLMClient().generate("What is the refund window?", chunks)


async def test_hallucinate_mode_fabricates_a_figure() -> None:
    chunks = [Chunk(id=1, document_id="d", content="There is an annual bonus.")]
    assert "10 percent" in await FakeLLMClient(hallucinate=True).generate("what percent?", chunks)


async def test_llm_admits_ignorance_with_no_context() -> None:
    assert "enough information" in (await FakeLLMClient().generate("q", [])).lower()


async def test_llm_failure_raises() -> None:
    with pytest.raises(LLMError):
        await FakeLLMClient(fail=True).generate("q", [])


def test_prompt_numbers_and_attributes_passages() -> None:
    chunks = [
        Chunk(id=1, document_id="doc-a", content="First passage."),
        Chunk(id=2, document_id="doc-b", content="Second passage."),
    ]
    prompt = build_prompt("What?", chunks)
    assert "[1]" in prompt and "[2]" in prompt
    assert "doc-a" in prompt and "doc-b" in prompt


def test_reasoning_blocks_are_stripped() -> None:
    """Open models emit <think> blocks that must not be fact-checked as claims."""
    raw = "<think>Maybe it is 10%, but I am unsure.</think>The bonus is set in December."
    assert _strip_reasoning(raw) == "The bonus is set in December."


def test_reasoning_stripper_leaves_normal_text_intact() -> None:
    assert _strip_reasoning("A plain answer.") == "A plain answer."


# --- factory / provider matrix ---------------------------------------------


def test_factory_returns_fake_by_default() -> None:
    assert isinstance(build_llm_client(Settings(llm_provider="fake")), FakeLLMClient)


@pytest.mark.parametrize("provider", ["openai", "openrouter", "together", "groq", "ollama"])
def test_openai_compatible_providers_share_one_client(provider: str) -> None:
    """Five providers, one implementation — they share a wire format."""
    client = build_llm_client(Settings(llm_provider=provider, llm_api_key="k"))
    assert isinstance(client, OpenAICompatibleClient)


def test_anthropic_gets_a_dedicated_client() -> None:
    """Anthropic differs in auth header, system field and response shape."""
    client = build_llm_client(Settings(llm_provider="anthropic", llm_api_key="k"))
    assert isinstance(client, AnthropicClient)


def test_custom_provider_requires_a_base_url() -> None:
    with pytest.raises(LLMError, match="BASE_URL"):
        build_llm_client(Settings(llm_provider="custom", llm_api_key="k"))


def test_custom_provider_with_base_url_and_model_works() -> None:
    client = build_llm_client(
        Settings(
            llm_provider="custom",
            llm_api_key="k",
            llm_base_url="https://my-gateway.internal/v1",
            llm_model="my-model",
        )
    )
    assert isinstance(client, OpenAICompatibleClient)


def test_live_provider_requires_a_key() -> None:
    with pytest.raises(LLMError, match="API_KEY"):
        build_llm_client(Settings(llm_provider="openai", llm_api_key=None))


def test_explicit_model_overrides_the_provider_default() -> None:
    settings = Settings(llm_provider="openai", llm_api_key="k", llm_model="gpt-4o")
    client = build_llm_client(settings)
    assert client._model == "gpt-4o"


def test_every_provider_has_a_default_model() -> None:
    """A missing default would surface as a confusing runtime 404."""
    for provider, (url, model) in PROVIDER_DEFAULTS.items():
        if provider == "custom":
            continue
        assert url and model, f"{provider} lacks a default"
