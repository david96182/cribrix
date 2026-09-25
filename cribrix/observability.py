"""Structured logging and per-stage timing.

A precision-oriented pipeline is only as useful as its auditability. Every
stage emits a structured event carrying the request id, so a single `jq` filter
reconstructs the full decision path for any request.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import structlog

from cribrix.schemas import PipelineTrace, StageTiming

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    """Generate a short, log-friendly correlation id."""
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str) -> None:
    """Bind the correlation id for the current async context."""
    _request_id.set(request_id)


def get_request_id() -> str:
    """Return the correlation id bound to the current async context."""
    return _request_id.get()


def _inject_request_id(
    _logger: object, _name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """structlog processor stamping every event with the active request id."""
    event_dict.setdefault("request_id", get_request_id())
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog + stdlib logging. Safe to call once at startup."""
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stderr,
    )

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _inject_request_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        # stderr, so CLI output on stdout (reports, --json) stays machine-readable.
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]


@contextmanager
def stage_timer(trace: PipelineTrace, stage: str) -> Iterator[None]:
    """Record the wall-clock duration of a pipeline stage into the trace.

    Uses ``perf_counter`` (monotonic) so NTP adjustments can't produce
    negative durations. Timing is recorded even if the block raises.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        trace.timings.append(StageTiming(stage=stage, duration_ms=round(elapsed_ms, 3)))
