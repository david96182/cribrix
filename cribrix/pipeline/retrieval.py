"""Stage 2 — Retrieval.

Thin seam between the pipeline and the persistence layer. It exists so the
orchestrator can depend on a `Retriever` Protocol rather than on SQLAlchemy,
which is what lets the entire pipeline be unit-tested without Postgres.

The embedder is likewise abstracted. Cribrix's thesis is about *filtering*
retrieved context, so the embedding model is deliberately swappable and the
default is a deterministic hash embedder — no model download, no API key, and
reproducible fixtures.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from cribrix.database import Database, cosine_search
from cribrix.observability import get_logger
from cribrix.schemas import Chunk

logger = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a dense vector."""

    @property
    def dimension(self) -> int:
        """Output dimensionality."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Fetches candidate chunks for a query."""

    async def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        """Return up to `top_k` candidate chunks, nearest-first."""
        ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (the "hashing trick").

    Not semantically strong — it captures lexical overlap, not meaning. That
    is an acceptable and deliberate tradeoff: it makes the repo runnable with
    zero external dependencies, and Cribrix's contribution is the triage and
    verification layers, not the encoder. Swap in a real model in production
    by implementing the `Embedder` Protocol.
    """

    def __init__(self, dimension: int = 1536) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Hash tokens into buckets, then L2-normalise for cosine distance."""
        vector = [0.0] * self._dimension
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0  # signed hashing reduces collision bias
            vector[index] += sign

        # Normalise so cosine distance is well-conditioned; zero vectors would
        # make every distance NaN, so fall back to a fixed unit vector.
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


class PgVectorRetriever:
    """Retriever backed by PostgreSQL + pgvector cosine search."""

    def __init__(self, database: Database, embedder: Embedder) -> None:
        self._db = database
        self._embedder = embedder

    async def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        """Embed the query and run an ANN search over the corpus."""
        embedding = await self._embedder.embed(query)
        async with self._db.session() as session:
            chunks = await cosine_search(session, embedding, top_k=top_k)
        logger.info("retrieval.completed", candidates=len(chunks), top_k=top_k)
        return chunks


class InMemoryVectorRetriever:
    """Exact cosine search over an in-memory corpus.

    Ranks exactly as ``PgVectorRetriever`` does with the same embedder (minus
    HNSW's approximation), so the evaluation harness retrieves the same
    candidates offline as the running stack does against Postgres.
    """

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self._chunks = list(chunks)
        self._embedder = embedder
        self._vectors: list[list[float]] | None = None

    async def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        if self._vectors is None:
            self._vectors = [await self._embedder.embed(c.content) for c in self._chunks]
        q = await self._embedder.embed(query)
        scored = []
        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            similarity = sum(a * b for a, b in zip(q, vector, strict=True))
            scored.append((1.0 - similarity, chunk.id, chunk))
        scored.sort(key=lambda item: (item[0], item[1]))  # id breaks ties deterministically
        return [c.model_copy(update={"distance": d}) for d, _, c in scored[:top_k]]


class InMemoryRetriever:
    """Test double holding a fixed corpus, ranked by token overlap.

    Lets the full pipeline be exercised in unit tests with no database.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)

    async def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        query_tokens = set(_TOKEN_RE.findall(query.lower()))

        def overlap(chunk: Chunk) -> int:
            return len(query_tokens & set(_TOKEN_RE.findall(chunk.content.lower())))

        ranked = sorted(self._chunks, key=overlap, reverse=True)
        return ranked[:top_k]
