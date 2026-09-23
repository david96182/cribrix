"""Render a `/query` response as a human-readable decision trace.

Reads a JSON response on stdin and prints what each pipeline stage decided.
The audit trail is the product here — a refusal is only convincing if you can
see *which* chunks were rejected and *how badly* each claim scored.

Usage::

    make ask Q="what is the refund window?"
    curl -s localhost:8000/query -d '{"query":"..."}' | python -m cribrix.evaluation.explain
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

BOLD, DIM, GREEN, YELLOW, RED, CYAN, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[36m",
    "\033[0m",
)

STATUS_COLOURS = {
    "ANSWERED": GREEN,
    "CHITCHAT": CYAN,
    "INSUFFICIENT_CONTEXT": YELLOW,
    "NO_DOCUMENTS": YELLOW,
    "UNGROUNDED": RED,
    "VERIFIER_UNAVAILABLE": RED,
}


def _colour() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def paint(text: str, code: str) -> str:
    """Colourise for a TTY, no-op when piped."""
    return f"{code}{text}{RESET}" if _colour() else text


def render(body: dict[str, Any]) -> None:
    """Print the answer plus a stage-by-stage explanation."""
    status = body.get("status", "UNKNOWN")
    colour = STATUS_COLOURS.get(status, BOLD)

    print()
    print(f"  {paint(status, colour)}   verified={body.get('verified')}")
    print(f"  {paint('answer', BOLD)}: {body.get('answer', '')}")

    trace = body.get("trace")
    if not trace:
        print()
        return

    intent = trace.get("intent")
    confidence = trace.get("intent_confidence")
    if intent:
        suffix = f" (confidence {confidence:.2f})" if isinstance(confidence, float) else ""
        print(f"\n  {paint('1 route', DIM)}      {intent}{suffix}")

    retrieved = trace.get("retrieved_count", 0)
    if retrieved:
        print(f"  {paint('2 retrieve', DIM)}   {retrieved} candidate chunks")

    scored = trace.get("scored_chunks") or []
    if scored:
        kept = trace.get("kept_count", 0)
        print(f"  {paint('3 triage', DIM)}     kept {kept}/{len(scored)}")
        for item in scored:
            mark = paint("KEEP", GREEN) if item["kept"] else paint("drop", DIM)
            text = item["chunk"]["content"].replace("\n", " ")[:62]
            print(f"      [{mark}] {item['relevance']:.2f}  {text}")

    verdicts = trace.get("claim_verdicts") or []
    if verdicts:
        groundedness = trace.get("groundedness")
        shown = f"{groundedness:.0%}" if isinstance(groundedness, float) else "n/a"
        print(f"  {paint('5 verify', DIM)}     groundedness {shown}")
        for verdict in verdicts:
            mark = paint("OK ", GREEN) if verdict["grounded"] else paint("BAD", RED)
            print(f"      [{mark}] p={verdict['probability']:.2f}  {verdict['claim'][:58]}")

    sources = body.get("sources") or []
    if sources:
        print(f"  {paint('sources', DIM)}      " + ", ".join(s["document_id"] for s in sources))

    timings = trace.get("timings") or []
    if timings:
        parts = " ".join(f"{t['stage']}={t['duration_ms']:.0f}ms" for t in timings)
        print(f"  {paint('timing', DIM)}       {parts}  total={trace.get('total_ms', 0):.0f}ms")

    for note in trace.get("notes") or []:
        print(paint(f"  note         {note}", DIM))
    print()


def main() -> int:
    """Read a response from stdin and render it."""
    raw = sys.stdin.read().strip()
    if not raw:
        print("No input. Is the API running? Try: make up && make seed", file=sys.stderr)
        return 1
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        # Usually means something other than Cribrix answered on the port.
        print(f"Expected JSON, got:\n{raw[:300]}", file=sys.stderr)
        return 1
    if "status" not in body:
        print(f"Unexpected response: {json.dumps(body)[:300]}", file=sys.stderr)
        return 1
    render(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
