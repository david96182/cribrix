"""HTTP-layer tests.

The database and both model clients are replaced with in-memory doubles via
dependency overrides, so the API surface is tested without any external
service.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from cribrix.clients.jev import FakeJevClient
from cribrix.clients.llm import FakeLLMClient
from cribrix.config import Settings, get_settings
from cribrix.database import Database
from cribrix.main import app, get_database, get_jev, get_llm, get_retriever
from cribrix.pipeline.retrieval import InMemoryRetriever
from cribrix.schemas import REFUSAL_STATUSES, AnswerStatus, Chunk


class _FakeDatabase(Database):
    """Database double that reports healthy and never opens a connection."""

    def __init__(self) -> None:
        pass

    async def ping(self) -> bool:
        return True


@pytest.fixture
async def client(chunks: list[Chunk], settings: Settings) -> AsyncIterator[AsyncClient]:
    """ASGI client with every external dependency overridden."""
    app.dependency_overrides[get_jev] = lambda: FakeJevClient()
    app.dependency_overrides[get_llm] = lambda: FakeLLMClient()
    app.dependency_overrides[get_retriever] = lambda: InMemoryRetriever(chunks)
    app.dependency_overrides[get_database] = lambda: _FakeDatabase()
    app.dependency_overrides[get_settings] = lambda: settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


async def test_query_returns_a_verified_answer(client: AsyncClient) -> None:
    resp = await client.post(
        "/query", json={"query": "What is the refund window for enterprise customers?"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == AnswerStatus.ANSWERED.value
    assert body["verified"] is True
    assert body["sources"]


async def test_refusal_is_a_200_not_an_error(client: AsyncClient) -> None:
    """A refusal is a successful pipeline outcome.

    Clients must branch on `status`, not on the HTTP code — returning 4xx/5xx
    here would make a correct, intentional decision look like a malfunction.
    """
    resp = await client.post(
        "/query", json={"query": "What was total revenue in fiscal year 2019?"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["verified"] is False
    assert body["status"] in {s.value for s in REFUSAL_STATUSES}


async def test_chitchat_is_handled(client: AsyncClient) -> None:
    resp = await client.post("/query", json={"query": "hello!"})
    assert resp.json()["status"] == AnswerStatus.CHITCHAT.value


async def test_trace_is_returned_by_default(client: AsyncClient) -> None:
    resp = await client.post("/query", json={"query": "refund window enterprise?"})
    trace = resp.json()["trace"]
    assert trace is not None
    assert trace["request_id"]
    assert trace["timings"]


async def test_trace_can_be_suppressed(client: AsyncClient) -> None:
    resp = await client.post("/query", json={"query": "refund window?", "include_trace": False})
    assert resp.json()["trace"] is None


@pytest.mark.parametrize("payload", [{}, {"query": ""}, {"query": "x" * 5000}])
async def test_invalid_payloads_are_rejected(client: AsyncClient, payload: dict) -> None:
    resp = await client.post("/query", json=payload)
    assert resp.status_code == 422


async def test_health_reports_each_dependency(client: AsyncClient) -> None:
    resp = await client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "up"
    assert body["jev"] == "up"
    assert body["llm"] == "up"


async def test_health_reports_degraded_not_error(chunks: list[Chunk]) -> None:
    """A degraded dependency must still yield 200 with detail.

    An orchestrator learns far more from "jev: down" than from a bare 500.
    """
    app.dependency_overrides[get_jev] = lambda: FakeJevClient(fail=True)
    app.dependency_overrides[get_llm] = lambda: FakeLLMClient()
    app.dependency_overrides[get_database] = lambda: _FakeDatabase()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        body = (await ac.get("/health")).json()

    assert body["status"] == "degraded"
    assert body["jev"] == "down"
    app.dependency_overrides.clear()


async def test_correlation_id_is_echoed(client: AsyncClient) -> None:
    resp = await client.post(
        "/query", json={"query": "hello"}, headers={"x-request-id": "trace-me-123"}
    )
    assert resp.headers["x-request-id"] == "trace-me-123"


async def test_correlation_id_is_generated_when_absent(client: AsyncClient) -> None:
    resp = await client.post("/query", json={"query": "hello"})
    assert resp.headers.get("x-request-id")


async def test_ingest_rejects_dimension_mismatch(client: AsyncClient, settings: Settings) -> None:
    """Catch this at the API boundary; pgvector's error is far less actionable."""
    resp = await client.post(
        "/ingest",
        json={"chunks": [{"document_id": "d1", "content": "text", "embedding": [0.1, 0.2, 0.3]}]},
    )

    assert resp.status_code == 422
    assert "dimension" in resp.json()["detail"].lower()


async def test_admin_reset_is_refused_outside_dev(chunks: list[Chunk]) -> None:
    """The destructive endpoint must be off unless explicitly in a dev environment."""
    app.dependency_overrides[get_database] = lambda: _FakeDatabase()
    app.dependency_overrides[get_settings] = lambda: Settings(env="production")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/admin/reset")
    app.dependency_overrides.clear()
    assert resp.status_code == 403


async def test_openapi_schema_is_generated(client: AsyncClient) -> None:
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    assert "/query" in resp.json()["paths"]
