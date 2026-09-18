"""Stage 5 — Groundedness verification (hallucination gate).

The last line of defence: nothing reaches the user unless the context supports
it.

Why atomic verification is the default
--------------------------------------
A single ``boolean(context, whole_answer)`` call is the obvious design and it
is wrong in practice. Real answers are multi-claim — four sentences where three
are grounded and one is invented. A holistic boolean collapses that to
``False`` and discards a mostly-correct answer, so the system feels broken and
users route around it.

Atomic mode splits the draft into sentence-level claims, verifies each
independently and concurrently, and passes when the grounded fraction clears
``groundedness_threshold``. That yields a *graded* signal the evaluator can
report on, instead of a single bit.

Refusal passthrough
-------------------
An honest "I don't have enough information" is not entailed by the context, so
a naive verifier rejects it — and the system replaces a correct refusal with a
generic error. Recognised refusal phrasings bypass the gate.
"""

from __future__ import annotations

import asyncio
import re

from cribrix.clients.jev import JevClient, JevError
from cribrix.observability import get_logger
from cribrix.schemas import Chunk, ClaimVerdict

logger = get_logger(__name__)

# Sentence splitter that tolerates common abbreviations and decimals
# (e.g. "approx. 47.3% in Q3") without shattering them into fragments.
_ABBREVIATIONS = r"(?<!\b[A-Z])(?<!\bapprox)(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bNo)"
_SENTENCE_SPLIT_RE = re.compile(rf"{_ABBREVIATIONS}(?<=[.!?])\s+(?=[A-Z0-9])")

_REFUSAL_PATTERNS = (
    "i don't have enough information",
    "i do not have enough information",
    "insufficient information",
    "the context does not",
    "the provided context does not",
    "i cannot answer",
    "i can't answer",
    "not enough context",
)

MIN_CLAIM_CHARS = 12
"""Fragments shorter than this ("Yes.", "Correct.") carry no verifiable
proposition; verifying them produces noise, so they are skipped."""


class VerificationResult:
    """Outcome of the groundedness gate."""

    __slots__ = ("groundedness", "passed", "reason", "verdicts")

    def __init__(
        self,
        passed: bool,
        verdicts: list[ClaimVerdict],
        groundedness: float | None,
        reason: str = "",
    ) -> None:
        self.passed = passed
        self.verdicts = verdicts
        self.groundedness = groundedness
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"VerificationResult(passed={self.passed}, "
            f"groundedness={self.groundedness}, reason={self.reason!r})"
        )


def is_refusal(answer: str) -> bool:
    """True if the draft is an explicit admission of insufficient information."""
    lowered = answer.lower()
    return any(pattern in lowered for pattern in _REFUSAL_PATTERNS)


def split_claims(answer: str) -> list[str]:
    """Decompose a draft into atomic, verifiable claims.

    Sentence splitting is a pragmatic approximation of claim extraction: it is
    cheap, deterministic, and captures most fabrications, which tend to arrive
    as whole invented sentences. A dedicated claim-extraction model would be
    the production upgrade; this keeps the dependency surface at zero.
    """
    cleaned = answer.strip()
    if not cleaned:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(cleaned)]
    return [p for p in parts if len(p) >= MIN_CLAIM_CHARS]


def build_context(chunks: list[Chunk]) -> str:
    """Concatenate surviving chunks into the evidence block for verification.

    Critically this uses the **triaged** chunks — the same evidence the
    generator saw. Verifying against the full retrieval set would let the model
    be credited for context it was never shown.
    """
    return "\n\n".join(chunk.content for chunk in chunks)


async def _verify_claim(
    jev: JevClient,
    semaphore: asyncio.Semaphore,
    context: str,
    claim: str,
    noul_threshold: float,
) -> ClaimVerdict:
    """Verify one claim; a failed call is treated as ungrounded (fail-closed).

    Jev's ``noul`` primitive returns a *probability*, not a boolean, so the
    threshold decision lives here rather than inside the client. Retaining the
    raw probability on the verdict is what lets the trace show *how* confident
    the rejection was instead of just that one happened.
    """
    async with semaphore:
        try:
            probability = await jev.grounded(source=context, claim=claim)
        except JevError:
            logger.warning("verification.claim_check_failed", exc_info=True)
            probability = 0.0
        return ClaimVerdict(
            claim=claim,
            grounded=probability >= noul_threshold,
            probability=round(probability, 4),
        )


async def verify_answer(
    jev: JevClient,
    chunks: list[Chunk],
    answer: str,
    *,
    mode: str = "atomic",
    groundedness_threshold: float = 1.0,
    noul_threshold: float = 0.5,
    fail_open: bool = False,
    max_concurrency: int = 16,
) -> VerificationResult:
    """Gate a draft answer on whether the context supports it.

    Args:
        jev: System-1 client.
        chunks: The triaged chunks that were given to the generator.
        answer: The draft response.
        mode: ``"atomic"`` (per-sentence) or ``"holistic"`` (single check).
        groundedness_threshold: Fraction of claims that must be grounded.
        noul_threshold: Minimum Noul probability for a claim to count as grounded.
        fail_open: If the verifier is unreachable, pass the answer through
            (flagged) instead of refusing. Default False = fail-closed.
        max_concurrency: Bound on simultaneous in-flight boolean calls.

    Returns:
        A ``VerificationResult`` carrying the pass/fail decision, per-claim
        verdicts and the groundedness ratio.
    """
    # An explicit refusal is honest and must not be punished by the gate.
    if is_refusal(answer):
        return VerificationResult(True, [], None, reason="refusal_passthrough")

    # No evidence means nothing can be grounded. Never fail open here: this is
    # precisely the situation the system exists to catch.
    if not chunks:
        return VerificationResult(False, [], 0.0, reason="no_context_to_verify_against")

    context = build_context(chunks)

    if mode == "holistic":
        try:
            probability = await jev.grounded(source=context, claim=answer)
        except JevError:
            return _on_verifier_error(fail_open)
        grounded = probability >= noul_threshold
        verdicts = [
            ClaimVerdict(claim=answer, grounded=grounded, probability=round(probability, 4))
        ]
        return VerificationResult(grounded, verdicts, 1.0 if grounded else 0.0, reason="holistic")

    claims = split_claims(answer)
    if not claims:
        # Nothing substantive was asserted (e.g. "Yes."). Nothing to fabricate.
        return VerificationResult(True, [], None, reason="no_verifiable_claims")

    semaphore = asyncio.Semaphore(max_concurrency)
    verdicts = await asyncio.gather(
        *(_verify_claim(jev, semaphore, context, claim, noul_threshold) for claim in claims)
    )

    grounded_count = sum(1 for v in verdicts if v.grounded)
    groundedness = grounded_count / len(verdicts)
    passed = groundedness >= groundedness_threshold

    logger.info(
        "verification.completed",
        claims=len(verdicts),
        grounded=grounded_count,
        groundedness=round(groundedness, 4),
        passed=passed,
    )
    return VerificationResult(passed, list(verdicts), groundedness, reason="atomic")


def _on_verifier_error(fail_open: bool) -> VerificationResult:
    """Apply the configured verifier-outage policy."""
    if fail_open:
        logger.error("verification.unavailable_failing_open")
        return VerificationResult(True, [], None, reason="verifier_unavailable_failed_open")
    logger.error("verification.unavailable_failing_closed")
    return VerificationResult(False, [], None, reason="verifier_unavailable")
