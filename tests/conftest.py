"""Shared pytest fixtures.

Everything here is dependency-free: the whole unit suite runs with no
Postgres, no network and no API keys. That is a direct consequence of the
pipeline depending on Protocols rather than concrete clients.
"""

from __future__ import annotations

import pytest

from cribrix.clients.jev import FakeJevClient
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
        # 0.65 rather than 0.7: Jev's Score primitive is ordinal, so normalised
        # values land on rubric steps (0, 1/3, 2/3, 1). A 0.7 threshold sits
        # just above the 2/3 step and would silently reject partial matches.
        # Threshold choice must respect the rubric's granularity.
        relevance_threshold=0.65,
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
