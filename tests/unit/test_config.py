"""Tests for configuration validation.

Config bugs are the worst class of production bug: silent, and only visible in
degraded output quality. These tests make misconfiguration loud and immediate.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cribrix.config import Settings, get_settings


def test_defaults_are_sane() -> None:
    s = Settings()
    assert 0.0 <= s.relevance_threshold <= 1.0
    assert s.retrieval_top_k >= s.max_chunks_to_llm, (
        "retrieval must be wider than what reaches the LLM, or triage is decorative"
    )
    assert s.fail_open_on_verifier_error is False, "default must be fail-closed"


def test_sync_driver_is_rejected() -> None:
    """A sync DSN would block the event loop; fail at startup, not under load."""
    with pytest.raises(ValidationError, match="asyncpg"):
        Settings(database_url="postgresql://u:p@localhost/db")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_threshold_must_be_a_probability(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(relevance_threshold=value)


def test_invalid_verification_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(verification_mode="vibes")  # type: ignore[arg-type]


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(log_level="CHATTY")


def test_log_level_is_normalised_to_upper_case() -> None:
    assert Settings(log_level="debug").log_level == "DEBUG"


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(retrieval_top_k=0)


def test_settings_are_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
