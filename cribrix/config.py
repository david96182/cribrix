"""Centralised, validated application configuration.

Every tunable in Cribrix lives here. There are deliberately **no magic numbers**
in the pipeline code: thresholds, fan-out limits and failure policies are all
declared in this module so they can be tuned per-corpus by the evaluation
harness rather than by editing source.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_ENVIRONMENTS = frozenset({"local", "test", "docker"})


class Settings(BaseSettings):
    """Runtime configuration, sourced from environment / `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="CRIBRIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database -----------------------------------------------------------
    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+asyncpg://cribrix:cribrix@localhost:5432/cribrix"),
        description="Async SQLAlchemy DSN. Must use the asyncpg driver.",
    )
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=5, ge=0, le=100)
    db_echo: bool = False

    # --- Embeddings ---------------------------------------------------------
    embedding_dim: int = Field(
        default=384,
        ge=8,
        le=16000,
        description=(
            "Dimension of the vector column. Changing this requires a re-index. "
            "384 suits the default HashingEmbedder and matches common small "
            "sentence encoders (e.g. all-MiniLM-L6-v2); set 1536 for "
            "text-embedding-3-small."
        ),
    )

    # --- Retrieval ----------------------------------------------------------
    retrieval_top_k: int = Field(
        default=20,
        ge=1,
        le=200,
        description=(
            "Retrieve wide. Recall is retrieval's job; precision is the cribrix's job. "
            "A tight top_k makes the triage stage decorative."
        ),
    )

    # --- Triage (the cribrix) -------------------------------------------------
    relevance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Chunks whose normalised relevance falls below this are discarded. "
            "Jev's Score is the probability-weighted mean of the rubric levels "
            "(0..3 for the 4-level rubric), divided by 3 here, so it is a "
            "continuous 0..1 value: 0.33 = 'same topic, different entity', "
            "0.67 = 'partially answers', 1.0 = 'directly answers'. Calibrate "
            "with `make eval-sweep` against recorded live answers."
        ),
    )
    evidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum Noul probability that a chunk *states* the requested "
            "information. Separates answerability from topical relevance. "
            "0.0 disables the check."
        ),
    )
    injection_max: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Chunks whose prompt-injection Noul exceeds this never reach the "
            "generator. A filter, not a security boundary."
        ),
    )
    chitchat_min_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Choice confidence required to skip retrieval as CHITCHAT. Anything "
            "less certain is searched: a wasted search is cheaper than an "
            "unanswered question."
        ),
    )
    min_chunks_required: int = Field(
        default=1,
        ge=0,
        description=(
            "If fewer than this many chunks survive triage, refuse instead of "
            "generating. Guards against the #1 hallucination cause: an LLM asked "
            "to answer with an empty context."
        ),
    )
    max_chunks_to_llm: int = Field(
        default=8,
        ge=1,
        le=100,
        description="Hard cap on chunks forwarded to the generator (cost + context window).",
    )

    # --- Verification -------------------------------------------------------
    verification_mode: Literal["atomic", "holistic"] = Field(
        default="atomic",
        description=(
            "'atomic' asks one Noul per sentence-level claim (all in a single "
            "request); 'holistic' asks one Noul about the whole draft. Both cost "
            "one request; atomic reports which claim failed."
        ),
    )
    groundedness_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of atomic claims that must be grounded for the answer to pass.",
    )
    fail_open_on_verifier_error: bool = Field(
        default=False,
        description=(
            "Verifier outage policy. False = fail-closed (refuse). This is a product "
            "decision, made explicit rather than left to an exception handler."
        ),
    )

    # --- Jev (System 1) -----------------------------------------------------
    jev_mode: Literal["fake", "live"] = Field(
        default="fake",
        description="'fake' uses the deterministic offline client; 'live' calls TypeSafe AI.",
    )
    jev_api_key: str | None = None
    jev_model: str | None = Field(
        default=None,
        description="Jev model name, e.g. 'jev-latest'. None uses the SDK default.",
    )
    jev_base_url: str | None = Field(
        default=None,
        description="Override the TypeSafe API root, e.g. to route via an AI gateway.",
    )
    jev_timeout_s: float = Field(default=15.0, gt=0)
    jev_max_concurrency: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Semaphore bound on concurrent Jev requests during triage.",
    )
    noul_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Noul returns the probability that a claim is supported. A claim "
            "counts as grounded at or above this value."
        ),
    )

    # --- LLM (System 2) -----------------------------------------------------
    llm_provider: Literal[
        "fake", "openai", "anthropic", "openrouter", "together", "groq", "ollama", "custom"
    ] = Field(
        default="fake",
        description=(
            "All providers except 'anthropic' and 'fake' speak the OpenAI "
            "chat-completions format and share one client implementation."
        ),
    )
    llm_api_key: str | None = None
    llm_model: str | None = Field(
        default=None, description="Model id. None falls back to the provider default."
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Override the provider base URL. Required when provider='custom'.",
    )
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Default 0.0: a pipeline judged on groundedness must be reproducible.",
    )
    llm_max_tokens: int = Field(default=500, ge=16, le=8192)

    # --- Ports / host ---------------------------------------------------------
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description=(
            "Host port the API is published on. Shared by docker compose, "
            "`make seed` and the host-side test helpers so one edit moves "
            "everything consistently."
        ),
    )
    db_port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="Host port PostgreSQL is published on (container side stays 5432).",
    )
    api_host: str = Field(
        default="127.0.0.1",
        description="Interface the published API port binds to.",
    )

    @property
    def api_base_url(self) -> str:
        """Base URL clients should use to reach this API on the host.

        Derived rather than separately configured, so the port can never drift
        between docker-compose and the scripts that talk to it.
        """
        host = "localhost" if self.api_host in {"0.0.0.0", "127.0.0.1", ""} else self.api_host
        return f"http://{host}:{self.api_port}"

    # --- App ----------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    env: str = Field(
        default="production",
        description=(
            "Deployment environment. Development conveniences (auto schema "
            "creation, the unauthenticated /admin/reset endpoint) are enabled "
            "only for 'local', 'test' and 'docker'. The default is the safe one: "
            "a deployment that forgets to set CRIBRIX_ENV gets production behaviour."
        ),
    )

    @property
    def is_dev(self) -> bool:
        """True in environments where destructive dev conveniences are allowed."""
        return self.env in DEV_ENVIRONMENTS

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, v: PostgresDsn) -> PostgresDsn:
        """Fail fast on a sync DSN rather than deadlocking the event loop later."""
        if "+asyncpg" not in str(v):
            raise ValueError(
                "database_url must use the asyncpg driver, "
                "e.g. postgresql+asyncpg://user:pass@host/db"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that FastAPI's `Depends(get_settings)` is effectively free and
    so tests can override via `get_settings.cache_clear()`.
    """
    return Settings()
