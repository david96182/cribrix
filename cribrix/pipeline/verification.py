"""Stage 5 — Groundedness verification (hallucination gate).

The last line of defence: nothing reaches the user unless the context supports
it.

Design
------
**Atomic, but one request.** The draft is split into sentence-level claims and
every claim becomes its own ``Noul`` question — all inside a *single* Jev
request whose state is the evidence. Questions are evaluated independently and
in parallel, so verifying ten claims costs roughly what verifying one does.

**Numbers are checked in code.** Jev's own documentation lists numeric
comparison as a weak spot. Every number in a claim must literally appear in the
evidence (after normalising "1,000", "99.95%", "five", "first"...). A claim
that introduces a figure the evidence never states is ungrounded regardless of
what the model says. This check can only make the gate stricter.

**No silent bypasses.**

* A draft is treated as a refusal only if *nothing but* refusal language
  remains once contrastive clauses are split off. "The context doesn't say,
  but it is 10%" is a refusal wrapper around a claim, and the claim is checked.
* Short fragments ("It is 10%.", "Yes.") are never skipped. They are verified
  with the user's question attached, so the model can judge what they assert.
* An empty draft fails.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from cribrix.clients.jev import JevClient, JevError
from cribrix.observability import get_logger
from cribrix.schemas import Chunk, ClaimVerdict

logger = get_logger(__name__)

# Sentence splitter that tolerates common abbreviations and decimals
# (e.g. "approx. 47.3% in Q3") without shattering them into fragments.
_ABBREVIATIONS = r"(?<!\b[A-Z])(?<!\bapprox)(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bNo)"
_SENTENCE_SPLIT_RE = re.compile(rf"{_ABBREVIATIONS}(?<=[.!?])\s+(?=[A-Z0-9])")

_REFUSAL_RE = re.compile(
    r"\b(?:"
    r"i (?:do not|don't) (?:have|know)(?: (?:enough|sufficient|that|this|the))?"
    r"(?: (?:information|context|detail))?"
    r"|(?:insufficient|not enough) (?:information|context)"
    r"|(?:there is |there's )?no information (?:is |was )?(?:available|provided|given|about|on)"
    r"|(?:is|are) not (?:publicly )?(?:specified|stated|provided|available)\b"
    r"|none of the (?:provided |given |available )?(?:context )?(?:passages?|sources?|documents?)"
    r"(?: passages?)? (?:mention|contain|include|state|specify|cover|say|provide)"
    r"|(?:the )?(?:provided |given |available )?"
    r"(?:context|passages?|sources?|documents?|information)(?: passages?)? "
    r"(?:does not|doesn't|do not|don't|contains? no) "
    r"(?:contain|specify|state|mention|say|include|provide|cover)?"
    r"|(?:is|are) not (?:specified|stated|mentioned|provided|given|included|publicly specified) "
    r"(?:in the (?:context|passages?|sources?|provided|available|given))?"
    r"|(?:i|we) (?:cannot|can't|am unable to|are unable to|am not able to) "
    r"(?:answer|determine|find|say|tell)"
    r"|(?:the )?(?:provided |available )?information does not (?:specify|state|mention|include)"
    r")\b",
    re.IGNORECASE,
)

# Markdown emphasis, list markers and "Answer:" labels LLMs wrap text in.
_MARKUP_RE = re.compile(r"[*_`#>]+|^\s*[-\u2022]\s+", re.MULTILINE)
_LABEL_RE = re.compile(r"^\s*(?:answer|response)\s*:\s*", re.MULTILINE | re.IGNORECASE)
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")

# Clause boundaries that commonly glue a claim onto a refusal.
_CONTRAST_RE = re.compile(
    r"\s*[;:]\s*|,?\s*\b(?:but|however|although|though|yet|nevertheless|nonetheless|"
    r"that said|instead)\b,?\s*",
    re.IGNORECASE,
)

MIN_CLAIM_CHARS = 12
"""Fragments shorter than this rarely carry a self-contained proposition, so
they are verified *with the question attached* rather than on their own."""

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50, "hundred": 100,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "once": 1, "twice": 2, "single": 1, "half": 50, "dozen": 12,
}  # fmt: skip


@dataclass
class VerificationResult:
    """Outcome of the groundedness gate."""

    passed: bool
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    groundedness: float | None = None
    reason: str = ""
    declined: bool = False
    """True when the draft was purely an admission that the context lacks the answer."""


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def split_claims(answer: str) -> list[str]:
    """Split a draft into sentence-level claims. Nothing is discarded."""
    cleaned = _LABEL_RE.sub("", _MARKUP_RE.sub("", answer)).strip()
    if not cleaned:
        return []
    parts: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(cleaned):
        parts.extend(p.strip() for p in _SENTENCE_SPLIT_RE.split(paragraph.strip()) if p.strip())
    return parts


def is_refusal(text: str) -> bool:
    """True if `text` contains refusal language."""
    return bool(_REFUSAL_RE.search(text))


def extract_claims(answer: str) -> tuple[list[str], bool]:
    """Separate refusal language from the claims that must be verified.

    Returns:
        ``(claims, had_refusal)``. A draft with ``had_refusal`` and no claims
        is a pure refusal.
    """
    claims: list[str] = []
    had_refusal = False
    for sentence in split_claims(answer):
        if not is_refusal(sentence):
            claims.append(sentence)
            continue
        had_refusal = True
        for part in _CONTRAST_RE.split(sentence):
            part = part.strip(" ,.")
            # Anything that is not itself refusal language is an assertion.
            if part and not is_refusal(part) and re.search(r"[A-Za-z0-9]", part):
                claims.append(part)
    return claims, had_refusal


def contextualise(claim: str, question: str | None) -> str:
    """Attach the question to fragments too short to stand alone."""
    if question and len(claim) < MIN_CLAIM_CHARS:
        return f"Asked '{question}', the answer was: {claim}"
    return claim


# ---------------------------------------------------------------------------
# Deterministic numeric check
# ---------------------------------------------------------------------------


def _normalise_number(raw: str) -> str:
    value = float(raw.replace(",", ""))
    return f"{value:g}"


def numbers_in(text: str, *, words: frozenset[str] | None = None) -> set[str]:
    """All numbers in `text`, normalised ("1,000" == "1000", "five" == "5").

    `words` restricts which number-words are recognised (default: all).
    """
    vocabulary = _WORD_NUMBERS.keys() if words is None else words
    found = {_normalise_number(m) for m in _NUMBER_RE.findall(text) if m.strip(",")}
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in vocabulary:
            found.add(f"{_WORD_NUMBERS[word]:g}")
    return found


# On the *claim* side only unambiguous cardinals count: "one of the options",
# "per second" or "a third party" are prose, and flagging them would make the
# gate over-refuse. The evidence side recognises every number word, so a claim
# saying "5 retries" is supported by evidence saying "five times".
_CLAIM_NUMBER_WORDS = frozenset(
    [
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "fifteen",
        "twenty",
        "thirty",
        "fifty",
        "hundred",
        "twice",
        "dozen",
    ]
)


def unsupported_numbers(claim: str, evidence: str) -> list[str]:
    """Numbers stated in `claim` that never appear in `evidence`."""
    claimed = numbers_in(claim, words=_CLAIM_NUMBER_WORDS)
    return sorted(claimed - numbers_in(evidence), key=float)


def build_context(chunks: list[Chunk]) -> str:
    """Concatenate surviving chunks into the evidence block for verification.

    Critically this uses the **triaged** chunks — the same evidence the
    generator saw. Verifying against the full retrieval set would let the model
    be credited for context it was never shown.
    """
    return "\n\n".join(chunk.content for chunk in chunks)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


async def verify_answer(
    jev: JevClient,
    chunks: list[Chunk],
    answer: str,
    *,
    question: str | None = None,
    mode: str = "atomic",
    groundedness_threshold: float = 1.0,
    noul_threshold: float = 0.5,
    fail_open: bool = False,
) -> VerificationResult:
    """Gate a draft answer on whether the context supports it.

    Args:
        jev: System-1 client.
        chunks: The triaged chunks that were given to the generator.
        answer: The draft response.
        question: The user's question, used to give short fragments meaning.
        mode: ``"atomic"`` (per-claim) or ``"holistic"`` (the whole draft as one claim).
        groundedness_threshold: Fraction of claims that must be grounded.
        noul_threshold: Minimum Noul probability for a claim to count as grounded.
        fail_open: If the verifier is unreachable, pass the answer through
            (flagged) instead of refusing. Default False = fail-closed.
    """
    claims, had_refusal = extract_claims(answer)

    if not claims:
        if had_refusal:
            # An honest, *pure* admission of ignorance is not a fabrication.
            return VerificationResult(True, reason="generator_declined", declined=True)
        return VerificationResult(False, groundedness=0.0, reason="empty_draft")

    # No evidence means nothing can be grounded. Never fail open here: this is
    # precisely the situation the system exists to catch.
    if not chunks:
        return VerificationResult(False, groundedness=0.0, reason="no_context_to_verify_against")

    context = build_context(chunks)
    if mode == "holistic":
        claims = [" ".join(claims)]
    to_check = [contextualise(c, question) for c in claims]

    try:
        result = await jev.verify_claims(context, to_check)
    except JevError:
        return _on_verifier_error(fail_open)
    if len(result.probabilities) != len(to_check):
        logger.error("verification.malformed_response")
        return _on_verifier_error(fail_open)

    verdicts: list[ClaimVerdict] = []
    for claim, probability in zip(claims, result.probabilities, strict=True):
        missing = unsupported_numbers(claim, context)
        verdicts.append(
            ClaimVerdict(
                claim=claim,
                grounded=probability >= noul_threshold and not missing,
                probability=round(probability, 4),
                unsupported_numbers=missing,
            )
        )

    grounded_count = sum(1 for v in verdicts if v.grounded)
    groundedness = grounded_count / len(verdicts)
    passed = groundedness >= groundedness_threshold

    logger.info(
        "verification.completed",
        mode=mode,
        claims=len(verdicts),
        grounded=grounded_count,
        groundedness=round(groundedness, 4),
        passed=passed,
    )
    return VerificationResult(passed, verdicts, groundedness, reason=mode)


def _on_verifier_error(fail_open: bool) -> VerificationResult:
    """Apply the configured verifier-outage policy."""
    if fail_open:
        logger.error("verification.unavailable_failing_open")
        return VerificationResult(True, reason="verifier_unavailable_failed_open")
    logger.error("verification.unavailable_failing_closed")
    return VerificationResult(False, reason="verifier_unavailable")


__all__: Sequence[str] = (
    "VerificationResult",
    "build_context",
    "contextualise",
    "extract_claims",
    "is_refusal",
    "numbers_in",
    "split_claims",
    "unsupported_numbers",
    "verify_answer",
)
