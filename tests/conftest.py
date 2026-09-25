"""Shared pytest fixtures.

Everything here is dependency-free: the whole unit suite runs with no
Postgres, no network and no API keys. That is a direct consequence of the
pipeline depending on Protocols rather than concrete clients.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from cribrix.clients.jev import (
    FakeJevClient,
    GroundingResult,
    JevError,
    PassageAssessment,
    RouteDecision,
)
from cribrix.clients.llm import FakeLLMClient
from cribrix.config import Settings
from cribrix.pipeline.orchestrator import RAGPipeline
from cribrix.pipeline.retrieval import InMemoryRetriever
from cribrix.schemas import Chunk


@pytest.fixture
def settings() -> Settings:
    """Default test configuration: strict thresholds, mock clients."""
    return Settings(
        database_url="postgresql+asyncpg://cribrix:cribrix@localhost:5432/cribrix",  # type: ignore[arg-type]
        relevance_threshold=0.5,
        evidence_threshold=0.5,
        min_chunks_required=1,
        retrieval_top_k=10,
        max_chunks_to_llm=5,
        verification_mode="atomic",
        groundedness_threshold=1.0,
        fail_open_on_verifier_error=False,
        jev_mode="fake",
        llm_provider="fake",
        env="test",
        log_json=False,
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    """Small, readable corpus covering several distinct topics."""
    return [
        Chunk(
            id=1,
            document_id="billing",
            content=(
                "Enterprise customers may request a full refund within 30 days of the invoice date."
            ),
            distance=0.12,
        ),
        Chunk(
            id=2,
            document_id="billing",
            content="Standard plans have a 14 day refund window.",
            distance=0.25,
        ),
        Chunk(
            id=3,
            document_id="security",
            content="Customer data is encrypted at rest using AES-256.",
            distance=0.71,
        ),
        Chunk(
            id=4,
            document_id="misc",
            content="The office cafeteria serves lunch between 12pm and 2pm.",
            distance=0.95,
        ),
    ]


@pytest.fixture
def jev() -> FakeJevClient:
    """Deterministic System-1 client."""
    return FakeJevClient()


@pytest.fixture
def llm() -> FakeLLMClient:
    """Deterministic, extractive System-2 client."""
    return FakeLLMClient()


@pytest.fixture
def pipeline(
    jev: FakeJevClient,
    llm: FakeLLMClient,
    chunks: list[Chunk],
    settings: Settings,
) -> RAGPipeline:
    """Fully wired pipeline over the in-memory corpus."""
    return RAGPipeline(
        jev=jev,
        llm=llm,
        retriever=InMemoryRetriever(chunks),
        settings=settings,
    )


class ScriptedJev:
    """Programmable Jev double for stage-level tests.

    Each operation is driven by a callable (or a fixed value), and every call
    is recorded so tests can assert on what the pipeline actually asked.
    """

    def __init__(
        self,
        *,
        route: RouteDecision | None = None,
        assess: Callable[[str, str], PassageAssessment] | None = None,
        verify: Callable[[str], float] | None = None,
        fail: set[str] | None = None,
    ) -> None:
        self._route = route or RouteDecision("SEARCH", 1.0, {"SEARCH": 1.0, "CHITCHAT": 0.0})
        self._assess = assess or (lambda q, p: PassageAssessment(1.0, 1.0, 1.0, 0.0))
        self._verify = verify or (lambda claim: 0.98)
        self._fail = fail or set()
        self.route_calls: list[tuple[str, dict[str, str]]] = []
        self.assessed: list[str] = []
        self.verify_calls: list[list[str]] = []

    async def route(self, message: str, options: dict[str, str]) -> RouteDecision:
        self.route_calls.append((message, options))
        if "route" in self._fail:
            raise JevError("route down")
        return self._route

    async def assess_passage(self, question: str, passage: str) -> PassageAssessment:
        self.assessed.append(passage)
        if "assess" in self._fail or passage in self._fail:
            raise JevError("assess down")
        return self._assess(question, passage)

    async def verify_claims(self, source: str, claims: Sequence[str]) -> GroundingResult:
        self.verify_calls.append(list(claims))
        if "verify" in self._fail:
            raise JevError("verify down")
        return GroundingResult([self._verify(c) for c in claims])

    async def health(self) -> bool:
        return not self._fail

    async def aclose(self) -> None:
        return None
