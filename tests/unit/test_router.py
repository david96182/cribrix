"""Unit tests for Stage 1 — intent routing."""

from __future__ import annotations

import pytest

from cribrix.clients.jev import ROUTING_CRITERIA, FakeJevClient, JevError
from cribrix.pipeline.router import route_intent
from cribrix.schemas import Intent


class _StubJev:
    """Jev stub returning a canned label, or raising."""

    def __init__(self, label: str | None = None, *, raises: bool = False) -> None:
        self._label = label
        self._raises = raises
        self.seen_options: dict[str, str] | None = None

    async def choice(self, context: object, options: dict[str, str]) -> tuple[str, float]:
        self.seen_options = options
        if self._raises:
            raise JevError("boom")
        assert self._label is not None
        return self._label, 0.9

    async def score(self, question: str, document: str) -> float:  # pragma: no cover
        return 0.0

    async def score_batch(self, question: str, documents: list[str]) -> list[float]:
        return [0.0] * len(documents)  # pragma: no cover

    async def grounded(self, source: str, claim: str) -> float:  # pragma: no cover
        return 1.0

    async def health(self) -> bool:  # pragma: no cover
        return not self._raises

    async def aclose(self) -> None:  # pragma: no cover
        return None


@pytest.mark.parametrize(
    "query",
    [
        "What is the refund policy for enterprise contracts?",
        "Explain how the encryption key rotation works",
        "Which document describes the SLA uptime guarantee?",
    ],
)
async def test_substantive_questions_route_to_search(jev: FakeJevClient, query: str) -> None:
    assert (await route_intent(jev, query))[0] is Intent.SEARCH


@pytest.mark.parametrize("query", ["hi", "hello there", "thanks!", "bye"])
async def test_greetings_route_to_chitchat(jev: FakeJevClient, query: str) -> None:
    assert (await route_intent(jev, query))[0] is Intent.CHITCHAT


async def test_router_passes_exactly_the_two_expected_options() -> None:
    """The option set is part of the contract with the classifier."""
    stub = _StubJev("SEARCH")
    await route_intent(stub, "anything")
    assert stub.seen_options == ROUTING_CRITERIA
    assert set(stub.seen_options) == {"SEARCH", "CHITCHAT"}
    # Criteria carry descriptions, not bare labels: the model is told what
    # each option *means*, which is what makes the live router accurate.
    assert all(isinstance(v, str) and v for v in stub.seen_options.values())


async def test_router_normalises_case_and_whitespace() -> None:
    """A well-behaved client shouldn't return this, but tolerate it anyway."""
    assert (await route_intent(_StubJev("  search  "), "q"))[0] is Intent.SEARCH


async def test_router_defaults_to_search_on_unknown_label() -> None:
    """An unrecognised label must not be allowed to skip retrieval."""
    assert (await route_intent(_StubJev("MAYBE_SEARCH"), "q"))[0] is Intent.SEARCH


async def test_router_defaults_to_search_on_client_error() -> None:
    """Fail towards retrieval.

    Misrouting CHITCHAT->SEARCH wastes a little compute. Misrouting
    SEARCH->CHITCHAT silently answers a real question with no grounding and no
    fact-check. The costs are asymmetric, so the default is.
    """
    assert (await route_intent(_StubJev(raises=True), "q?"))[0] is Intent.SEARCH


async def test_router_is_deterministic(jev: FakeJevClient) -> None:
    """Same input, same decision — required for reproducible evaluation."""
    query = "What is the refund window?"
    first, _ = await route_intent(jev, query)
    for _ in range(5):
        assert (await route_intent(jev, query))[0] is first
