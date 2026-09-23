#!/usr/bin/env python3
"""Load the demo corpus into a running Cribrix instance and take it for a spin.

Usage::

    make up && make seed

The target URL comes from ``CRIBRIX_API_PORT`` / ``CRIBRIX_API_HOST`` in
``.env`` (override with ``CRIBRIX_BASE_URL``), so moving the port in one place
moves it everywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

from cribrix.config import get_settings
from cribrix.evaluation.demo_corpus import DEMO_CORPUS, DEMO_QUESTIONS
from cribrix.pipeline.retrieval import HashingEmbedder

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)


def _supports_colour() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def paint(text: str, colour: str) -> str:
    """Colourise for a TTY, no-op when piped."""
    return f"{colour}{text}{RESET}" if _supports_colour() else text


def fail(message: str, *hints: str) -> int:
    """Print an actionable error and return a non-zero exit code."""
    print(paint(f"\n  ERROR  {message}", RED), file=sys.stderr)
    for hint in hints:
        print(f"         {hint}", file=sys.stderr)
    return 1


async def preflight(client: httpx.AsyncClient, base_url: str) -> str | None:
    """Verify that a healthy *Cribrix* API is answering on `base_url`.

    Returns an error message, or None when everything checks out.
    """
    try:
        response = await client.get("/health")
    except httpx.ConnectError:
        return f"nothing is listening on {base_url}"
    except httpx.HTTPError as exc:
        return f"could not reach {base_url}: {exc}"

    if response.status_code == 404:
        return (
            f"something is running on {base_url}, but it has no /health endpoint "
            "- it is probably not Cribrix"
        )

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        # This is the exact failure this script used to die on.
        server = response.headers.get("server", "unknown")
        return (
            f"{base_url} replied with {content_type or 'an unknown content type'} "
            f"(server: {server}), not JSON - another application is using this port"
        )

    body = response.json()
    if "database" not in body or "jev" not in body:
        return f"{base_url} returned unexpected JSON: {str(body)[:120]}"

    if body.get("database") != "up":
        return (
            "the API is running but its database is unreachable "
            f"(health reports database={body.get('database')!r})"
        )
    return None


async def seed(client: httpx.AsyncClient, *, dimension: int, reset: bool) -> int | str:
    """Embed and upload the demo corpus. Returns rows inserted, or an error."""
    stats = (await client.get("/stats")).json()
    existing = int(stats.get("chunks", 0))

    if existing and not reset:
        print(
            paint(
                f"  corpus already has {existing} chunks - skipping ingest "
                "(use --reset to replace)",
                YELLOW,
            )
        )
        return 0

    if existing and reset:
        response = await client.post("/admin/reset")
        if response.status_code >= 400:
            return f"reset failed: HTTP {response.status_code} {response.text[:160]}"
        print(f"  cleared {response.json().get('deleted', 0)} existing chunks")

    embedder = HashingEmbedder(dimension=dimension)
    payload = {
        "chunks": [
            {
                "document_id": chunk.document_id,
                "content": chunk.content,
                "embedding": await embedder.embed(chunk.content),
                "metadata": {"seeded": True},
            }
            for chunk in DEMO_CORPUS
        ]
    }
    response = await client.post("/ingest", json=payload)
    if response.status_code >= 400:
        return f"ingest failed: HTTP {response.status_code} {response.text[:200]}"
    return int(response.json()["inserted"])


async def tour(client: httpx.AsyncClient) -> None:
    """Run the guided questions and show what the pipeline decided."""
    print(f"\n{paint('Guided tour', BOLD)} - each question targets a different branch:\n")
    for item in DEMO_QUESTIONS:
        try:
            response = await client.post(
                "/query", json={"query": item.question, "include_trace": True}
            )
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            print(paint(f"  ? {item.question}\n      request failed: {exc}", RED))
            continue

        status = body["status"]
        matched = status in item.expectation
        mark = paint("PASS", GREEN) if matched else paint("DIFF", YELLOW)
        trace = body.get("trace") or {}

        print(f"  [{mark}] {item.question}")
        print(f"         expected : {item.expectation}")
        print(f"         actual   : {paint(status, BOLD)}")
        print(
            f"         pipeline : retrieved={trace.get('retrieved_count', 0)} "
            f"kept={trace.get('kept_count', 0)} "
            f"groundedness={trace.get('groundedness')}"
        )
        print(f"         answer   : {body['answer'][:110]}")
        print(paint(f"         why      : {item.why}", DIM))
        print()


async def main_async(args: argparse.Namespace) -> int:
    """Preflight, seed, then tour."""
    settings = get_settings()
    base_url = os.getenv("CRIBRIX_BASE_URL") or settings.api_base_url

    print(f"\n{paint('Cribrix demo seeder', BOLD)}")
    print(f"  target    : {base_url}")
    print(f"  corpus    : {len(DEMO_CORPUS)} chunks")
    print(f"  dimension : {settings.embedding_dim}")

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        problem = await preflight(client, base_url)
        if problem:
            return fail(
                problem,
                "",
                "Checks:",
                "  1. Is the stack up?          docker compose ps",
                "  2. Which port is published?  grep CRIBRIX_API_PORT .env",
                f"  3. Who else is on the port?  lsof -nP -iTCP:{settings.api_port} -sTCP:LISTEN",
                "",
                "If the port is taken, edit CRIBRIX_API_PORT in .env and re-run `make up`.",
            )
        print(paint("  health    : ok", GREEN))

        result = await seed(client, dimension=settings.embedding_dim, reset=args.reset)
        if isinstance(result, str):
            return fail(result)
        if result:
            print(paint(f"  ingested  : {result} chunks", GREEN))

        if not args.no_tour:
            await tour(client)

        print(f"{paint('Next:', BOLD)}")
        print(f"  Interactive docs   {base_url}/docs")
        print("  Ask your own       make ask Q='what is the API rate limit?'")
        print("  Compare pipelines  make scenarios")
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Seed the Cribrix demo corpus")
    parser.add_argument(
        "--reset", action="store_true", help="Delete existing chunks before seeding."
    )
    parser.add_argument("--no-tour", action="store_true", help="Seed without querying.")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
