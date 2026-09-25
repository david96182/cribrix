"""Async persistence layer: pgvector schema, engine lifecycle, and cosine search.

Key decisions
-------------
**Cosine over L2.** Text embeddings from modern encoders are direction-bearing;
magnitude mostly encodes token count. ``vector_cosine_ops`` is the correct
operator class, and the ``<=>`` operator returns cosine *distance* in [0, 2].

**HNSW over IVFFlat.** HNSW needs no training step, so it works on an empty
table and stays correct as rows are inserted incrementally — which is what an
ingest-as-you-go service actually does. IVFFlat requires a populated table at
index-build time and silently degrades if you skip the rebuild.

**Distance returned raw.** The search function returns the pgvector distance
untouched and lets the domain layer derive similarity. Converting in SQL would
bake a cosine-specific assumption into the storage layer.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Integer, MetaData, String, Text, delete, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cribrix.config import Settings, get_settings
from cribrix.schemas import Chunk

logger = logging.getLogger(__name__)

# Explicit naming convention so Alembic autogenerate produces stable,
# human-readable migration names instead of database-assigned defaults.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base with a deterministic constraint naming convention."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _embedding_dim() -> int:
    """Resolve the vector dimension at class-definition time."""
    return get_settings().embedding_dim


class DocumentChunk(Base):
    """A single embedded fragment of a source document."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(_embedding_dim()), nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    __table_args__ = (
        # HNSW ANN index tuned for cosine distance.
        #   m=16              -> graph connectivity; the pgvector default, good general tradeoff
        #   ef_construction=64 -> build-time accuracy; higher = better recall, slower build
        # Query-time recall is tuned separately via `hnsw.ef_search` (see cosine_search).
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def to_domain(self, distance: float = 0.0) -> Chunk:
        """Map the ORM row onto the immutable domain model."""
        return Chunk(
            id=self.id,
            document_id=self.document_id,
            content=self.content,
            metadata=dict(self.meta or {}),
            distance=distance,
        )


def column_dimension() -> int:
    """Return the vector dimension the mapped `embedding` column was built with.

    The ORM column is the authoritative dimension — it is what Postgres will
    actually enforce. Validating against `Settings` instead would let a config
    change pass validation and then fail deep inside the driver.
    """
    column_type = DocumentChunk.__table__.c.embedding.type
    return int(column_type.dim)  # type: ignore[attr-defined]  # pgvector.Vector


class Database:
    """Owns the async engine and session factory for the process lifetime."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        """Create the engine and session factory. Idempotent."""
        if self._engine is not None:
            return
        self._engine = create_async_engine(
            str(self._settings.database_url),
            echo=self._settings.db_echo,
            pool_size=self._settings.db_pool_size,
            max_overflow=self._settings.db_max_overflow,
            pool_pre_ping=True,  # survive Postgres restarts / idle connection reaping
        )
        self._sessionmaker = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )
        logger.info("database engine initialised")

    async def disconnect(self) -> None:
        """Dispose the connection pool."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None
            logger.info("database engine disposed")

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Database.connect() must be called before use")
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session inside a transaction, rolling back on error."""
        if self._sessionmaker is None:
            raise RuntimeError("Database.connect() must be called before use")
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # -- schema -------------------------------------------------------------

    async def create_schema(self) -> None:
        """Enable pgvector and create tables/indexes.

        Convenient for local dev and tests. Production should use Alembic —
        `CREATE EXTENSION` requires superuser, and HNSW builds are expensive
        enough to warrant an explicit, reviewed migration.
        """
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("schema ready (pgvector enabled, tables + HNSW index created)")

    async def drop_schema(self) -> None:
        """Drop all tables. Test teardown only."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def ping(self) -> bool:
        """Return True if the database answers a trivial query."""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception:
            logger.warning("database ping failed", exc_info=True)
            return False
        return True


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


async def cosine_search(
    session: AsyncSession,
    embedding: Sequence[float],
    *,
    top_k: int = 20,
    document_ids: Sequence[str] | None = None,
    ef_search: int | None = None,
) -> list[Chunk]:
    """Return the `top_k` nearest chunks by cosine distance.

    Args:
        session: Active async session.
        embedding: Query vector; length must equal the column dimension.
        top_k: Number of candidates to return. Retrieve wide — the triage
            stage is responsible for precision.
        document_ids: Optional restriction to a subset of source documents.
        ef_search: HNSW query-time search breadth. Must be >= top_k or recall
            silently collapses. Defaults to ``max(40, top_k * 2)``.

    Returns:
        Chunks ordered nearest-first, each carrying its raw cosine distance.
    """
    if top_k <= 0:
        return []

    # The single most common pgvector misconfiguration: ef_search < LIMIT.
    # Postgres will happily return fewer / worse results without warning.
    effective_ef = ef_search if ef_search is not None else max(40, top_k * 2)
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(effective_ef)}"))

    distance = DocumentChunk.embedding.cosine_distance(embedding).label("distance")
    stmt = select(DocumentChunk, distance)
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(list(document_ids)))
    stmt = stmt.order_by(distance).limit(top_k)

    result = await session.execute(stmt)
    return [row.DocumentChunk.to_domain(distance=float(row.distance)) for row in result]


async def insert_chunks(
    session: AsyncSession,
    rows: Sequence[tuple[str, str, Sequence[float], dict[str, Any]]],
) -> int:
    """Bulk-insert (document_id, content, embedding, metadata) tuples.

    Returns:
        Number of rows inserted.
    """
    if not rows:
        return 0
    # Derive the expected dimension from the mapped column, not from global
    # settings: the column is the actual source of truth, and reading config
    # here would let a settings change silently disagree with the live schema.
    expected = column_dimension()
    objects = []
    for doc_id, content, embedding, meta in rows:
        if len(embedding) != expected:
            raise ValueError(
                f"embedding for document {doc_id!r} has dimension {len(embedding)}, "
                f"expected {expected}"
            )
        objects.append(
            DocumentChunk(document_id=doc_id, content=content, embedding=list(embedding), meta=meta)
        )
    session.add_all(objects)
    await session.flush()
    return len(objects)


async def count_chunks(session: AsyncSession) -> int:
    """Total number of chunks in the corpus."""
    result = await session.execute(select(func.count()).select_from(DocumentChunk))
    return int(result.scalar_one())


async def delete_all_chunks(session: AsyncSession) -> int:
    """Remove every chunk. Used by the demo seeder's --reset flag.

    Uses DELETE rather than TRUNCATE so it participates in the surrounding
    transaction and rolls back cleanly on error.
    """
    result = await session.execute(delete(DocumentChunk))
    # CursorResult exposes rowcount; the generic Result stub does not.
    return int(getattr(result, "rowcount", 0) or 0)


async def delete_document(session: AsyncSession, document_id: str) -> int:
    """Delete every chunk belonging to `document_id`; returns rows removed.

    A single bulk DELETE: loading each row into the session and deleting it
    individually would be one round-trip per chunk.
    """
    stmt = delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
    result = await session.execute(stmt)
    return int(getattr(result, "rowcount", 0) or 0)


# Process-wide instance, wired into FastAPI's lifespan.
db = Database()
