"""Integration tests against a live PostgreSQL + pgvector instance.

Skipped unless a database is reachable, so `pytest` stays green on a clean
checkout. Run the real thing with:

    docker compose up -d db
    pytest -m integration
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text as sa_text
from sqlalchemy.exc import SQLAlchemyError

from cribrix.config import Settings
from cribrix.database import (
    Database,
    column_dimension,
    cosine_search,
    count_chunks,
    delete_document,
    insert_chunks,
)
from cribrix.pipeline.retrieval import HashingEmbedder, PgVectorRetriever

pytestmark = pytest.mark.integration

# The mapped column is the source of truth for dimensionality. Hardcoding a
# different value here (and trying to override settings after class definition)
# cannot work: the Vector(N) column is bound when the model class is created.
DIM = column_dimension()


@pytest.fixture
async def database():
    """Provision a clean schema per test, tearing it down afterwards.

    Skips cleanly when no PostgreSQL is reachable, so `pytest` stays green on
    a fresh checkout with no Docker running.
    """
    url = os.getenv(
        "CRIBRIX_TEST_DATABASE_URL", "postgresql+asyncpg://cribrix:cribrix@localhost:5432/cribrix"
    )
    db = Database(Settings(database_url=url, env="test"))  # type: ignore[arg-type]
    db.connect()
    try:
        await db.create_schema()
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        await db.disconnect()
        pytest.skip(f"PostgreSQL+pgvector not available: {exc}")
    except Exception as exc:
        # e.g. asyncpg.InvalidPasswordError: something else owns the port.
        await db.disconnect()
        if type(exc).__module__.startswith("asyncpg"):
            pytest.skip(f"PostgreSQL at {url} rejected the connection: {exc}")
        raise

    # Start from an empty corpus regardless of what previous runs left behind.
    async with db.session() as session:
        await session.execute(sa_text("TRUNCATE TABLE document_chunks RESTART IDENTITY"))

    yield db

    await db.drop_schema()
    await db.disconnect()


def _vec(seed: float) -> list[float]:
    """Deterministic unit-ish vector for testing."""
    return [seed] + [0.0] * (DIM - 1)


async def test_schema_creation_is_idempotent(database: Database) -> None:
    await database.create_schema()
    assert await database.ping() is True


async def test_insert_and_count(database: Database) -> None:
    async with database.session() as session:
        inserted = await insert_chunks(
            session,
            [
                ("doc1", "first chunk", _vec(1.0), {"page": 1}),
                ("doc1", "second chunk", _vec(0.9), {"page": 2}),
            ],
        )
        assert inserted == 2

    async with database.session() as session:
        assert await count_chunks(session) == 2


async def test_insert_rejects_wrong_dimension(database: Database) -> None:
    async with database.session() as session:
        with pytest.raises(ValueError, match="dimension"):
            await insert_chunks(session, [("doc", "text", [0.1, 0.2], {})])


async def test_cosine_search_orders_by_proximity(database: Database) -> None:
    near = [1.0] + [0.0] * (DIM - 1)
    far = [0.0, 1.0] + [0.0] * (DIM - 2)

    async with database.session() as session:
        await insert_chunks(
            session, [("doc", "near chunk", near, {}), ("doc", "far chunk", far, {})]
        )

    async with database.session() as session:
        results = await cosine_search(session, near, top_k=2)

    assert [c.content for c in results] == ["near chunk", "far chunk"]
    assert results[0].distance < results[1].distance
    assert results[0].vector_similarity > results[1].vector_similarity


async def test_cosine_distance_is_in_expected_range(database: Database) -> None:
    v = [1.0] + [0.0] * (DIM - 1)
    async with database.session() as session:
        await insert_chunks(session, [("doc", "identical", v, {})])

    async with database.session() as session:
        results = await cosine_search(session, v, top_k=1)

    # An identical vector should have ~0 cosine distance.
    assert results[0].distance == pytest.approx(0.0, abs=1e-6)


async def test_top_k_limits_results(database: Database) -> None:
    async with database.session() as session:
        await insert_chunks(
            session, [("doc", f"chunk {i}", _vec(1.0 - i / 100), {}) for i in range(10)]
        )

    async with database.session() as session:
        assert len(await cosine_search(session, _vec(1.0), top_k=3)) == 3


async def test_search_on_empty_table_returns_empty(database: Database) -> None:
    async with database.session() as session:
        assert await cosine_search(session, _vec(1.0), top_k=5) == []


async def test_document_id_filter(database: Database) -> None:
    async with database.session() as session:
        await insert_chunks(
            session,
            [("doc-a", "alpha", _vec(1.0), {}), ("doc-b", "beta", _vec(0.99), {})],
        )

    async with database.session() as session:
        results = await cosine_search(session, _vec(1.0), top_k=10, document_ids=["doc-b"])

    assert [c.document_id for c in results] == ["doc-b"]


async def test_metadata_roundtrips_through_jsonb(database: Database) -> None:
    meta = {"page": 7, "section": "billing", "tags": ["refund", "policy"]}
    async with database.session() as session:
        await insert_chunks(session, [("doc", "text", _vec(1.0), meta)])

    async with database.session() as session:
        results = await cosine_search(session, _vec(1.0), top_k=1)

    assert results[0].metadata == meta


async def test_delete_document_removes_all_its_chunks(database: Database) -> None:
    async with database.session() as session:
        await insert_chunks(
            session,
            [
                ("doc-a", "a1", _vec(1.0), {}),
                ("doc-a", "a2", _vec(0.9), {}),
                ("doc-b", "b1", _vec(0.8), {}),
            ],
        )

    async with database.session() as session:
        assert await delete_document(session, "doc-a") == 2

    async with database.session() as session:
        assert await count_chunks(session) == 1


async def test_transaction_rolls_back_on_error(database: Database) -> None:
    """A failed unit of work must leave no partial writes behind."""
    with pytest.raises(RuntimeError):
        async with database.session() as session:
            await insert_chunks(session, [("doc", "should vanish", _vec(1.0), {})])
            raise RuntimeError("simulated failure")

    async with database.session() as session:
        assert await count_chunks(session) == 0


async def test_retriever_end_to_end(database: Database) -> None:
    """Embed, store, and retrieve through the real retriever."""
    embedder = HashingEmbedder(dimension=DIM)
    texts = [
        "Enterprise customers receive a refund within 30 days of invoice.",
        "The cafeteria serves lunch between noon and two.",
    ]
    async with database.session() as session:
        await insert_chunks(session, [("doc", t, await embedder.embed(t), {}) for t in texts])

    retriever = PgVectorRetriever(database, embedder)
    results = await retriever.retrieve("enterprise refund invoice", top_k=2)

    assert len(results) == 2
    assert "refund" in results[0].content.lower(), "lexically closest chunk should rank first"
