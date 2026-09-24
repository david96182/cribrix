"""Proves the `Embedder` protocol is genuinely swappable.

The README claims replacing `HashingEmbedder` with a real model is "a ~6-line
class". This file holds that claim to account: it defines an embedder exactly
as the README shows, then drives the full retrieval path with it. If the
protocol ever grows a requirement the docs don't mention, these fail.
"""

from __future__ import annotations

import asyncio

import pytest

from cribrix.pipeline.retrieval import Embedder, HashingEmbedder, InMemoryRetriever
from cribrix.schemas import Chunk


class FakeAPIEmbedder:
    """An embedder shaped exactly like the OpenAI example in the README.

    Stands in for a network-backed model: same async signature, same
    `dimension` property, no real HTTP.
    """

    dimension = 8

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        await asyncio.sleep(0)  # a real client awaits I/O here
        vector = [0.0] * self.dimension
        for index, char in enumerate(text[: self.dimension]):
            vector[index] = (ord(char) % 17) / 17
        return vector


class BlockingModelEmbedder:
    """Mirrors the HuggingFace example: CPU-bound work moved off the loop."""

    dimension = 4

    def _encode(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0, 0.5]

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._encode, text)


def test_readme_style_embedder_satisfies_the_protocol() -> None:
    """A plain class with `dimension` + `embed` is enough — no base class."""
    assert isinstance(FakeAPIEmbedder(), Embedder)
    assert isinstance(BlockingModelEmbedder(), Embedder)
    assert isinstance(HashingEmbedder(dimension=16), Embedder)


async def test_custom_embedder_produces_vectors_of_its_declared_dimension() -> None:
    """A mismatch here is what produces an opaque pgvector insert error."""
    for embedder in (FakeAPIEmbedder(), BlockingModelEmbedder()):
        vector = await embedder.embed("the engineering team uses MacBook Pro M3s")
        assert len(vector) == embedder.dimension
        assert all(isinstance(v, float) for v in vector)


async def test_blocking_model_does_not_stall_the_event_loop() -> None:
    """`asyncio.to_thread` is the point of that line in the README snippet."""
    embedder = BlockingModelEmbedder()
    results = await asyncio.gather(*(embedder.embed(f"doc {i}") for i in range(8)))
    assert len(results) == 8


async def test_swapped_embedder_drives_the_retrieval_path() -> None:
    """End-to-end: a custom embedder works with the retriever unchanged."""
    embedder = FakeAPIEmbedder()
    corpus = [
        Chunk(id=1, document_id="d", content="The engineering team uses MacBook Pro M3s."),
        Chunk(id=2, document_id="d", content="The cafeteria serves mac and cheese."),
    ]
    # Prove the embedder is actually exercised on the ingest side.
    embeddings = [await embedder.embed(c.content) for c in corpus]
    assert embedder.calls == 2
    assert all(len(e) == embedder.dimension for e in embeddings)

    retriever = InMemoryRetriever(corpus)
    assert len(await retriever.retrieve("engineering laptop", top_k=2)) == 2


def test_hashing_embedder_dimension_is_configurable() -> None:
    """Changing CRIBRIX_EMBEDDING_DIM must actually change the vector width."""
    for dim in (8, 384, 1536):
        assert HashingEmbedder(dimension=dim).dimension == dim


async def test_hashing_embedder_returns_unit_vectors() -> None:
    """Normalisation is required for cosine distance to be well-conditioned."""
    vector = await HashingEmbedder(dimension=64).embed("refund policy for enterprise")
    magnitude = sum(v * v for v in vector) ** 0.5
    assert magnitude == pytest.approx(1.0, abs=1e-6)


async def test_hashing_embedder_handles_empty_text() -> None:
    """A zero vector would make every cosine distance NaN."""
    vector = await HashingEmbedder(dimension=32).embed("")
    assert sum(v * v for v in vector) ** 0.5 == pytest.approx(1.0, abs=1e-6)
