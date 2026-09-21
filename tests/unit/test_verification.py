"""Unit tests for Stage 5 — the groundedness gate."""

from __future__ import annotations

from cribrix.clients.jev import JevError
from cribrix.pipeline.verification import (
    is_refusal,
    split_claims,
    verify_answer,
)
from cribrix.schemas import Chunk


class _ScriptedJev:
    """Jev stub returning preset booleans, keyed by claim substring."""

    def __init__(
        self,
        verdicts: dict[str, bool] | None = None,
        *,
        default: bool = True,
        raises: bool = False,
    ) -> None:
        self._verdicts = verdicts or {}
        self._default = default
        self._raises = raises
        self.claims_seen: list[str] = []

    async def choice(  # pragma: no cover
        self, context: object, options: dict[str, str]
    ) -> tuple[str, float]:
        return next(iter(options)), 1.0

    async def score(self, question: str, document: str) -> float:  # pragma: no cover
        return 1.0

    async def score_batch(  # pragma: no cover
        self, question: str, documents: list[str]
    ) -> list[float]:
        return [1.0] * len(documents)

    async def grounded(self, source: str, claim: str) -> float:
        """Return a probability, mirroring the real Noul primitive."""
        if self._raises:
            raise JevError("verifier down")
        self.claims_seen.append(claim)
        for needle, verdict in self._verdicts.items():
            if needle in claim:
                return 0.98 if verdict else 0.02
        return 0.98 if self._default else 0.02

    async def health(self) -> bool:  # pragma: no cover
        return not self._raises

    async def aclose(self) -> None:  # pragma: no cover
        return None


CTX = [Chunk(id=1, document_id="d", content="The refund window is 30 days for enterprise.")]


# --- claim splitting -------------------------------------------------------


def test_split_claims_separates_sentences() -> None:
    claims = split_claims("The window is 30 days. Refunds take 10 business days.")
    assert len(claims) == 2


def test_split_claims_does_not_break_on_decimals() -> None:
    """'47.3%' must not be shattered into two claims."""
    claims = split_claims("Revenue grew by 47.3 percent this year in the region.")
    assert len(claims) == 1


def test_split_claims_drops_trivial_fragments() -> None:
    """'Yes.' asserts nothing verifiable and would only add noise."""
    assert split_claims("Yes.") == []


def test_split_claims_on_empty_input() -> None:
    assert split_claims("   ") == []


# --- refusal detection -----------------------------------------------------


def test_recognises_refusal_phrasings() -> None:
    assert is_refusal("I don't have enough information to answer that.")
    assert is_refusal("The provided context does not mention revenue.")


def test_normal_answer_is_not_a_refusal() -> None:
    assert not is_refusal("The refund window is 30 days.")


# --- the gate --------------------------------------------------------------


async def test_fully_grounded_answer_passes() -> None:
    result = await verify_answer(
        _ScriptedJev(default=True), CTX, "The refund window is 30 days for enterprise."
    )
    assert result.passed is True
    assert result.groundedness == 1.0


async def test_fully_ungrounded_answer_is_blocked() -> None:
    result = await verify_answer(
        _ScriptedJev(default=False), CTX, "Revenue in 2019 was 4.2 billion dollars."
    )
    assert result.passed is False
    assert result.groundedness == 0.0


async def test_partially_grounded_answer_is_blocked_at_strict_threshold() -> None:
    """Three true claims plus one fabrication must still fail at threshold 1.0.

    This is the scenario a single holistic boolean handles poorly and the
    reason atomic verification is the default.
    """
    answer = (
        "The refund window is 30 days. Requests go through the billing portal. "
        "Processing takes 10 business days. Revenue rose 47 percent in Zanzibar."
    )
    jev = _ScriptedJev({"Zanzibar": False}, default=True)

    result = await verify_answer(jev, CTX, answer, groundedness_threshold=1.0)

    assert result.passed is False
    assert result.groundedness == 0.75
    assert sum(1 for v in result.verdicts if not v.grounded) == 1


async def test_partially_grounded_answer_passes_at_relaxed_threshold() -> None:
    """The threshold is a real dial, not decoration."""
    answer = (
        "The refund window is 30 days. Requests go through the billing portal. "
        "Processing takes 10 business days. Revenue rose 47 percent in Zanzibar."
    )
    jev = _ScriptedJev({"Zanzibar": False}, default=True)

    result = await verify_answer(jev, CTX, answer, groundedness_threshold=0.7)

    assert result.passed is True
    assert result.groundedness == 0.75


async def test_every_claim_is_checked_individually() -> None:
    jev = _ScriptedJev(default=True)
    answer = "First claim is here. Second claim is here. Third claim is here."

    await verify_answer(jev, CTX, answer)

    assert len(jev.claims_seen) == 3


async def test_refusal_bypasses_the_gate() -> None:
    """An honest refusal is not entailed by the context, but it is correct.

    Without this passthrough the system would replace a correct "I don't know"
    with a generic error — strictly worse for the user.
    """
    jev = _ScriptedJev(default=False)

    result = await verify_answer(jev, CTX, "I don't have enough information to answer that.")

    assert result.passed is True
    assert result.reason == "refusal_passthrough"
    assert jev.claims_seen == []


async def test_empty_context_always_fails_regardless_of_fail_open() -> None:
    """Nothing can be grounded in nothing; never fail open here."""
    result = await verify_answer(
        _ScriptedJev(default=True), [], "Some confident claim about things.", fail_open=True
    )
    assert result.passed is False
    assert result.reason == "no_context_to_verify_against"


async def test_verifier_outage_fails_closed_by_default() -> None:
    result = await verify_answer(
        _ScriptedJev(raises=True), CTX, "A claim.", mode="holistic", fail_open=False
    )
    assert result.passed is False
    assert result.reason == "verifier_unavailable"


async def test_verifier_outage_can_fail_open_when_configured() -> None:
    """Availability-over-safety is a legitimate product choice — made explicit."""
    result = await verify_answer(
        _ScriptedJev(raises=True), CTX, "A claim.", mode="holistic", fail_open=True
    )
    assert result.passed is True
    assert result.reason == "verifier_unavailable_failed_open"


async def test_per_claim_failure_is_treated_as_ungrounded() -> None:
    """In atomic mode a failed check must never be optimistically passed."""
    result = await verify_answer(
        _ScriptedJev(raises=True), CTX, "A sufficiently long claim here.", mode="atomic"
    )
    assert result.passed is False
    assert result.groundedness == 0.0


async def test_holistic_mode_makes_exactly_one_call() -> None:
    jev = _ScriptedJev(default=True)
    answer = "First claim is here. Second claim is here. Third claim is here."

    result = await verify_answer(jev, CTX, answer, mode="holistic")

    assert len(jev.claims_seen) == 1
    assert result.passed is True


async def test_answer_with_no_verifiable_claims_passes() -> None:
    result = await verify_answer(_ScriptedJev(default=False), CTX, "Yes.")
    assert result.passed is True
    assert result.reason == "no_verifiable_claims"
