"""Unit tests for Stage 1 — intent routing."""

from __future__ import annotations

import pytest

from cribrix.clients.jev import ROUTING_CRITERIA, FakeJevClient, RouteDecision
from cribrix.pipeline.router import route_intent
from cribrix.schemas import Intent
from tests.conftest import ScriptedJev


def _stub(label: str, confidence: float = 0.95) -> ScriptedJev:
    return ScriptedJev(route=RouteDecision(label, confidence, {}))


@pytest.mark.parametrize(
    "query",
    [
        "What is the refund policy for enterprise contracts?",
        "Explain how the encryption key rotation works",
        "Which document describes the SLA uptime guarantee?",
        "office wifi password",
    ],
)
async def test_substantive_questions_route_to_search(jev: FakeJevClient, query: str) -> None:
    assert (await route_intent(jev, query))[0] is Intent.SEARCH


@pytest.mark.parametrize("query", ["hi", "hello there", "thanks!", "bye"])
async def test_greetings_route_to_chitchat(jev: FakeJevClient, query: str) -> None:
    assert (await route_intent(jev, query))[0] is Intent.CHITCHAT


async def test_router_passes_exactly_the_two_expected_options() -> None:
    stub = _stub("SEARCH")
    await route_intent(stub, "anything")
    _, options = stub.route_calls[0]
    assert options == ROUTING_CRITERIA
    assert set(options) == {"SEARCH", "CHITCHAT"}
    # Criteria carry descriptions, not bare labels.
    assert all(isinstance(v, str) and v for v in options.values())


async def test_confident_chitchat_skips_retrieval() -> None:
    assert await route_intent(_stub("CHITCHAT", 0.95), "hi") == (Intent.CHITCHAT, 0.95)


async def test_uncertain_chitchat_is_searched_instead() -> None:
    """Confidence-gated routing: CHITCHAT is only taken when the model is sure."""
    intent, confidence = await route_intent(
        _stub("CHITCHAT", 0.55), "thanks, what about refunds", chitchat_min_confidence=0.8
    )
    assert intent is Intent.SEARCH
    assert confidence == 0.55


async def test_low_confidence_search_is_still_search() -> None:
    """The gate only guards the cheap exit; SEARCH needs no confidence."""
    assert (await route_intent(_stub("SEARCH", 0.1), "q"))[0] is Intent.SEARCH


async def test_router_normalises_case_and_whitespace() -> None:
    assert (await route_intent(_stub("  search  "), "q"))[0] is Intent.SEARCH


async def test_router_defaults_to_search_on_unknown_label() -> None:
    assert (await route_intent(_stub("MAYBE_SEARCH"), "q"))[0] is Intent.SEARCH


async def test_router_defaults_to_search_on_client_error() -> None:
    """Fail towards retrieval: the costs of misrouting are asymmetric."""
    assert await route_intent(ScriptedJev(fail={"route"}), "q?") == (Intent.SEARCH, 0.0)


async def test_router_is_deterministic(jev: FakeJevClient) -> None:
    query = "What is the refund window?"
    first, _ = await route_intent(jev, query)
    for _ in range(5):
        assert (await route_intent(jev, query))[0] is first
