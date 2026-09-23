"""Tests for port configuration, the demo corpus and the seeder's preflight.

The preflight tests exist because of a real bug: `make seed` died with a raw
`JSONDecodeError` when an unrelated nginx application was listening on the
configured port. Misconfiguration must produce a sentence, not a stack trace.
"""

from __future__ import annotations

import httpx
import pytest

from cribrix.config import Settings
from cribrix.evaluation.demo_corpus import DEMO_CORPUS, DEMO_QUESTIONS
from cribrix.evaluation.explain import render

# ---------------------------------------------------------------------------
# Port configuration
# ---------------------------------------------------------------------------


def test_api_base_url_follows_the_configured_port() -> None:
    """One setting must drive compose, the seeder and the test helpers alike."""
    assert Settings(api_port=8008).api_base_url == "http://localhost:8008"


def test_bind_all_interfaces_still_yields_a_usable_client_url() -> None:
    """0.0.0.0 is a bind address, not something a client can connect to."""
    assert Settings(api_host="0.0.0.0", api_port=9000).api_base_url == "http://localhost:9000"


def test_explicit_host_is_preserved() -> None:
    settings = Settings(api_host="cribrix.internal", api_port=80)
    assert settings.api_base_url == "http://cribrix.internal:80"


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_out_of_range_ports_are_rejected(port: int) -> None:
    with pytest.raises(ValueError):
        Settings(api_port=port)


def test_db_port_is_independent_of_the_api_port() -> None:
    settings = Settings(api_port=8008, db_port=55432)
    assert (settings.api_port, settings.db_port) == (8008, 55432)


# ---------------------------------------------------------------------------
# Demo corpus
# ---------------------------------------------------------------------------


def test_corpus_is_substantial_enough_to_be_interesting() -> None:
    """A two-chunk corpus cannot demonstrate triage."""
    assert len(DEMO_CORPUS) >= 15


def test_chunk_ids_are_unique() -> None:
    ids = [c.id for c in DEMO_CORPUS]
    assert len(ids) == len(set(ids))


def test_corpus_spans_several_documents() -> None:
    assert len({c.document_id for c in DEMO_CORPUS}) >= 6


def test_corpus_contains_the_near_miss_distractor_pairs() -> None:
    """Tiered policies are what make triage non-trivial.

    If every chunk were about a distinct topic, cosine similarity alone would
    suffice and the sieve would be decorative.
    """
    text = " ".join(c.content.lower() for c in DEMO_CORPUS)
    assert "engineering team uses macbook pro" in text
    assert "marketing team uses macbook air" in text  # same-vocabulary distractor
    assert "mac and cheese" in text  # lexical noise
    assert "enterprise" in text and "standard and pro plans" in text  # tier pair


def test_corpus_has_a_deliberate_gap_for_the_hallucination_trap() -> None:
    """The bonus is mentioned; the percentage never is."""
    bonus = [c for c in DEMO_CORPUS if "bonus" in c.content.lower()]
    assert bonus, "the corpus must mention a bonus"
    assert not any("%" in c.content for c in bonus)


def test_every_demo_question_is_documented() -> None:
    assert len(DEMO_QUESTIONS) >= 5
    for item in DEMO_QUESTIONS:
        assert item.question.strip() and item.expectation.strip() and item.why.strip()


def test_demo_questions_cover_answers_refusals_and_chitchat() -> None:
    """A tour that only shows successes would prove nothing."""
    expectations = " ".join(q.expectation for q in DEMO_QUESTIONS)
    assert "ANSWERED" in expectations
    assert "INSUFFICIENT_CONTEXT" in expectations
    assert "CHITCHAT" in expectations


# ---------------------------------------------------------------------------
# Seeder preflight
# ---------------------------------------------------------------------------


def _client(handler: object) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_preflight_accepts_a_healthy_api() -> None:
    from scripts.seed import preflight

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "database": "up", "jev": "up"})

    async with _client(handler) as client:
        assert await preflight(client, "http://test") is None


async def test_preflight_detects_a_foreign_html_app() -> None:
    """The exact bug this replaces: nginx on the port, JSONDecodeError in the face."""
    from scripts.seed import preflight

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="<html>Odoo</html>", headers={"content-type": "text/html", "server": "nginx"}
        )

    async with _client(handler) as client:
        problem = await preflight(client, "http://test")

    assert problem is not None
    assert "not JSON" in problem
    assert "nginx" in problem


async def test_preflight_detects_a_json_api_that_is_not_cribrix() -> None:
    from scripts.seed import preflight

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hello": "world"})

    async with _client(handler) as client:
        problem = await preflight(client, "http://test")

    assert problem is not None and "unexpected JSON" in problem


async def test_preflight_reports_a_missing_health_route() -> None:
    from scripts.seed import preflight

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    async with _client(handler) as client:
        problem = await preflight(client, "http://test")

    assert problem is not None and "not Cribrix" in problem


async def test_preflight_reports_a_closed_port() -> None:
    from scripts.seed import preflight

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with _client(handler) as client:
        problem = await preflight(client, "http://test")

    assert problem is not None and "nothing is listening" in problem


async def test_preflight_reports_a_degraded_database() -> None:
    """The API can be up while its database is not; say which is broken."""
    from scripts.seed import preflight

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "degraded", "database": "down", "jev": "up"})

    async with _client(handler) as client:
        problem = await preflight(client, "http://test")

    assert problem is not None and "database" in problem


# ---------------------------------------------------------------------------
# Explain renderer
# ---------------------------------------------------------------------------


def test_render_handles_a_full_trace(capsys: pytest.CaptureFixture[str]) -> None:
    render(
        {
            "status": "ANSWERED",
            "verified": True,
            "answer": "The rate limit is 1000 requests per minute.",
            "sources": [{"document_id": "api-reference"}],
            "trace": {
                "intent": "SEARCH",
                "intent_confidence": 0.97,
                "retrieved_count": 20,
                "kept_count": 1,
                "scored_chunks": [
                    {"kept": True, "relevance": 1.0, "chunk": {"content": "rate limit 1000"}},
                    {"kept": False, "relevance": 0.1, "chunk": {"content": "cafeteria"}},
                ],
                "claim_verdicts": [
                    {"grounded": True, "probability": 0.99, "claim": "rate limit is 1000"}
                ],
                "groundedness": 1.0,
                "timings": [{"stage": "triage", "duration_ms": 12.0}],
                "total_ms": 42.0,
                "notes": ["verification: atomic"],
            },
        }
    )
    out = capsys.readouterr().out
    assert "ANSWERED" in out
    assert "KEEP" in out and "drop" in out
    assert "p=0.99" in out
    assert "api-reference" in out


def test_render_handles_a_response_without_a_trace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`include_trace=false` must not crash the formatter."""
    render({"status": "CHITCHAT", "verified": True, "answer": "Hello.", "trace": None})
    assert "CHITCHAT" in capsys.readouterr().out
