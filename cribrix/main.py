"""FastAPI application: HTTP surface for the Cribrix pipeline.

Wiring lives here and nowhere else. Clients, the embedder and the retriever are
constructed once during the lifespan and injected via `Depends`, which keeps
the pipeline itself free of framework and I/O concerns — and therefore
unit-testable without a running server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from cribrix import __version__
from cribrix.clients.jev import JevClient, build_jev_client
from cribrix.clients.llm import LLMClient, build_llm_client
from cribrix.config import Settings, get_settings
from cribrix.database import (
    Database,
    count_chunks,
    db,
    delete_all_chunks,
    insert_chunks,
)
from cribrix.observability import configure_logging, get_logger, new_request_id, set_request_id
from cribrix.pipeline.orchestrator import RAGPipeline
from cribrix.pipeline.retrieval import Embedder, HashingEmbedder, PgVectorRetriever, Retriever
from cribrix.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build long-lived dependencies on startup; release them on shutdown."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    db.connect()
    # Convenience for local dev and the containerised demo. In production this
    # belongs in a reviewed Alembic migration, not in application startup.
    if settings.is_dev:
        try:
            await db.create_schema()
        except Exception:
            logger.error("startup.schema_creation_failed", exc_info=True)

    embedder: Embedder = HashingEmbedder(dimension=settings.embedding_dim)
    app.state.settings = settings
    app.state.database = db
    app.state.embedder = embedder
    app.state.jev = build_jev_client(settings)
    app.state.llm = build_llm_client(settings)
    app.state.retriever = PgVectorRetriever(db, embedder)

    if not settings.is_dev:
        logger.info(
            "startup.production_mode", note="/admin/reset disabled; schema not auto-created"
        )
    logger.info(
        "startup.complete",
        env=settings.env,
        jev_mode=settings.jev_mode,
        llm_provider=settings.llm_provider,
        threshold=settings.relevance_threshold,
        verification_mode=settings.verification_mode,
    )
    try:
        yield
    finally:
        await app.state.jev.aclose()
        await app.state.llm.aclose()
        await db.disconnect()
        logger.info("shutdown.complete")


app = FastAPI(
    title="Cribrix",
    version=__version__,
    summary="A precision-first RAG orchestrator: filters before it generates, "
    "verifies before it answers.",
    description=(
        "Cribrix wraps a conventional LLM in a System-1 decision layer (Jev). "
        "Retrieved context is scored and filtered before generation, and every "
        "generated claim is fact-checked against that context before the answer "
        "is released. Unsupported answers are refused rather than returned."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Bind a correlation id to every request and echo it back to the client."""
    request_id = request.headers.get("x-request-id") or new_request_id()
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_jev(request: Request) -> JevClient:
    """Resolve the System-1 client from application state."""
    return request.app.state.jev  # type: ignore[no-any-return]


def get_llm(request: Request) -> LLMClient:
    """Resolve the System-2 generator from application state."""
    return request.app.state.llm  # type: ignore[no-any-return]


def get_retriever(request: Request) -> Retriever:
    """Resolve the retriever from application state."""
    return request.app.state.retriever  # type: ignore[no-any-return]


def get_database(request: Request) -> Database:
    """Resolve the database handle from application state."""
    return request.app.state.database  # type: ignore[no-any-return]


def get_embedder(request: Request) -> Embedder:
    """Resolve the embedder from application state."""
    return request.app.state.embedder  # type: ignore[no-any-return]


def get_pipeline(
    jev: Annotated[JevClient, Depends(get_jev)],
    llm: Annotated[LLMClient, Depends(get_llm)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RAGPipeline:
    """Assemble a pipeline instance for the current request.

    Construction is cheap (it only binds references), so a per-request instance
    keeps the pipeline stateless and free of cross-request leakage.
    """
    return RAGPipeline(jev=jev, llm=llm, retriever=retriever, settings=settings)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Run the full precision RAG pipeline",
    responses={
        200: {
            "description": (
                "Always 200 on a successful pipeline run. A refusal is a valid "
                "outcome, not an HTTP error — inspect `status` and `verified`."
            )
        }
    },
)
async def query_endpoint(
    payload: QueryRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> QueryResponse:
    """Route, retrieve, triage, generate and verify.

    Returns 200 even when the system refuses: "I will not answer that
    unreliably" is a successful, intentional outcome of the pipeline, and
    clients should branch on `status` rather than on an HTTP code.
    """
    return await pipeline.run(payload.query, include_trace=payload.include_trace)


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert pre-embedded chunks into the corpus",
)
async def ingest_endpoint(
    payload: IngestRequest,
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestResponse:
    """Bulk-insert chunks whose embeddings were computed upstream."""
    bad = [c.document_id for c in payload.chunks if len(c.embedding) != settings.embedding_dim]
    if bad:
        raise HTTPException(
            status_code=422,  # literal: the named constant differs across Starlette versions
            detail=(
                f"embedding dimension mismatch for documents {sorted(set(bad))}; "
                f"expected {settings.embedding_dim}"
            ),
        )
    rows = [(c.document_id, c.content, c.embedding, c.metadata) for c in payload.chunks]
    async with database.session() as session:
        inserted = await insert_chunks(session, rows)
    logger.info("ingest.completed", inserted=inserted)
    return IngestResponse(inserted=inserted)


@app.post(
    "/ingest/text",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert raw text, embedding it server-side",
)
async def ingest_text_endpoint(
    document_id: str,
    contents: list[str],
    database: Annotated[Database, Depends(get_database)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> IngestResponse:
    """Convenience ingest that embeds with the configured embedder."""
    rows: list[tuple[str, str, Sequence[float], dict[str, Any]]] = [
        (document_id, body, await embedder.embed(body), {}) for body in contents
    ]
    async with database.session() as session:
        inserted = await insert_chunks(session, rows)
    return IngestResponse(inserted=inserted)


@app.get("/health", response_model=HealthResponse, summary="Liveness and dependency probe")
async def health_endpoint(
    database: Annotated[Database, Depends(get_database)],
    jev: Annotated[JevClient, Depends(get_jev)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> HealthResponse:
    """Report the status of every downstream dependency.

    Never raises: a health endpoint that 500s tells an orchestrator far less
    than one that reports which specific dependency is degraded.
    """
    db_ok = await database.ping()
    jev_ok = await jev.health()
    llm_ok = await llm.health()
    overall = "ok" if (db_ok and jev_ok and llm_ok) else "degraded"
    return HealthResponse(
        status=overall,
        database="up" if db_ok else "down",
        jev="up" if jev_ok else "down",
        llm="up" if llm_ok else "down",
        version=__version__,
    )


@app.post(
    "/admin/reset",
    summary="Delete every chunk in the corpus",
    responses={403: {"description": "Refused outside local/test/docker environments."}},
)
async def reset_endpoint(
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, int]:
    """Truncate the corpus. Intended for demos and local iteration.

    Guarded by environment rather than left to a deployment checklist: a
    destructive, unauthenticated endpoint that exists in production is an
    incident waiting to happen.
    """
    if not settings.is_dev:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"/admin/reset is disabled when CRIBRIX_ENV={settings.env!r}",
        )
    async with database.session() as session:
        deleted = await delete_all_chunks(session)
    logger.info("admin.reset", deleted=deleted)
    return {"deleted": deleted}


@app.get("/stats", summary="Corpus size")
async def stats_endpoint(
    database: Annotated[Database, Depends(get_database)],
) -> dict[str, int]:
    """Return the number of indexed chunks."""
    async with database.session() as session:
        return {"chunks": await count_chunks(session)}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert unexpected errors into a stable JSON envelope.

    Deliberately leaks no exception detail to the caller; the stack trace goes
    to the structured log, correlated by request id.
    """
    logger.error("unhandled_exception", path=request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "request_id": request.headers.get("x-request-id", "-"),
        },
    )
