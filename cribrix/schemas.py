"""Public API contracts and internal pipeline data structures.

The response models are intentionally verbose: a precision-focused RAG system's
main selling point is that it can *explain why it answered the way it did*.
Every stage emits its decision into the trace so the caller (and the evaluator)
can audit the reasoning.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Intent(StrEnum):
    """Routing decision produced by the System-1 classifier."""

    SEARCH = "SEARCH"
    CHITCHAT = "CHITCHAT"


class AnswerStatus(StrEnum):
    """Terminal outcome of a pipeline run.

    Distinguishing the refusal reasons is what makes the system debuggable:
    "no documents in the corpus" and "documents found but none relevant" are
    completely different failures with completely different fixes.
    """

    ANSWERED = "ANSWERED"
    """Draft generated and passed verification."""

    CHITCHAT = "CHITCHAT"
    """Conversational turn; no retrieval performed."""

    NO_DOCUMENTS = "NO_DOCUMENTS"
    """Retrieval returned zero rows — the corpus is empty or filtered out."""

    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    """Chunks retrieved, but none cleared the relevance threshold."""

    UNGROUNDED = "UNGROUNDED"
    """A draft was produced but failed fact-checking; withheld from the user."""

    VERIFIER_UNAVAILABLE = "VERIFIER_UNAVAILABLE"
    """Verifier errored and policy is fail-closed."""


REFUSAL_MESSAGE = "Could not generate a reliable response with the current information."
"""Single canonical refusal string, so the contract is stable for clients."""


# ---------------------------------------------------------------------------
# Internal pipeline structures
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A retrieved document fragment, pre-triage."""

    model_config = ConfigDict(frozen=True)

    id: int
    document_id: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    distance: float = Field(
        default=0.0,
        description="Raw pgvector cosine distance in [0, 2]. Lower is closer.",
    )

    @property
    def vector_similarity(self) -> float:
        """Cosine similarity in [-1, 1], derived from the stored distance."""
        return 1.0 - self.distance


class ScoredChunk(BaseModel):
    """A chunk after the triage stage has assigned it a semantic relevance score."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    relevance: float = Field(ge=0.0, le=1.0, description="Jev semantic relevance score.")
    kept: bool = Field(description="Whether this chunk cleared the relevance threshold.")


class ClaimVerdict(BaseModel):
    """Verification outcome for a single atomic claim extracted from the draft."""

    model_config = ConfigDict(frozen=True)

    claim: str
    grounded: bool
    probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Raw Noul probability behind the verdict. Retained so the trace can "
            "show how confident a rejection was, not merely that one occurred."
        ),
    )


# ---------------------------------------------------------------------------
# Observability payloads
# ---------------------------------------------------------------------------


class StageTiming(BaseModel):
    """Wall-clock duration of one pipeline stage."""

    stage: str
    duration_ms: float


class PipelineTrace(BaseModel):
    """Structured audit trail of a single request.

    This is the feature that turns a demo into a tool: the caller can see the
    routing decision, every relevance score, and every claim-level verdict.
    """

    request_id: str
    intent: Intent | None = None
    intent_confidence: float | None = Field(
        default=None, description="Router confidence in the intent decision."
    )
    retrieved_count: int = 0
    scored_chunks: list[ScoredChunk] = Field(default_factory=list)
    kept_count: int = 0
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    groundedness: float | None = Field(
        default=None,
        description="Fraction of atomic claims found grounded. None if not verified.",
    )
    timings: list[StageTiming] = Field(default_factory=list)
    total_ms: float = 0.0
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP API contracts
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Inbound user query."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"query": "What is the refund window for enterprise plans?"}]
        }
    )

    query: str = Field(min_length=1, max_length=4000, description="The user's question.")
    include_trace: bool = Field(
        default=True,
        description="Return the full per-stage audit trail alongside the answer.",
    )


class SourceRef(BaseModel):
    """A citation for a chunk that actually survived triage and fed the answer."""

    chunk_id: int
    document_id: str
    relevance: float
    excerpt: str


class QueryResponse(BaseModel):
    """Outbound answer envelope."""

    status: AnswerStatus
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    verified: bool = Field(
        default=False,
        description="True only if the answer passed the groundedness gate.",
    )
    trace: PipelineTrace | None = None


class IngestChunk(BaseModel):
    """One chunk to write into the corpus."""

    document_id: str = Field(min_length=1, max_length=256)
    content: str = Field(min_length=1)
    embedding: list[float] = Field(description="Must match settings.embedding_dim.")
    metadata: dict[str, object] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Bulk ingest payload."""

    chunks: list[IngestChunk] = Field(min_length=1, max_length=1000)


class IngestResponse(BaseModel):
    """Result of a bulk ingest."""

    inserted: int


class HealthResponse(BaseModel):
    """Liveness / readiness probe payload."""

    status: str
    database: str
    jev: str
    llm: str
    version: str
