"""Unit tests for Stage 5 — the groundedness gate."""

from __future__ import annotations

import pytest

from cribrix.pipeline.verification import (
    extract_claims,
    is_refusal,
    numbers_in,
    split_claims,
    unsupported_numbers,
    verify_answer,
)
from cribrix.schemas import Chunk
from tests.conftest import ScriptedJev

CTX = [Chunk(id=1, document_id="d", content="The refund window is 30 days for enterprise.")]
BONUS = [
    Chunk(
        id=1,
        document_id="comp",
        content="The bonus percentage is decided by the board every December.",
    )
]


def _jev(bad: tuple[str, ...] = (), *, default: float = 0.98, **kw: object) -> ScriptedJev:
    """Claims containing any `bad` substring score 0.02; everything else `default`."""

    def verify(claim: str) -> float:
        return 0.02 if any(b in claim for b in bad) else default

    return ScriptedJev(verify=verify, **kw)  # type: ignore[arg-type]


# --- claim splitting -------------------------------------------------------


def test_split_claims_separates_sentences() -> None:
    assert len(split_claims("The window is 30 days. Refunds take 10 business days.")) == 2


def test_split_claims_does_not_break_on_decimals() -> None:
    """'47.3%' must not be shattered into two claims."""
    assert len(split_claims("Revenue grew by 47.3 percent this year in the region.")) == 1


def test_split_claims_keeps_short_fragments() -> None:
    """Short fragments are where a bare fabricated figure hides. Never drop them."""
    assert split_claims("It is 10%.") == ["It is 10%."]
    assert split_claims("Yes.") == ["Yes."]


def test_split_claims_on_empty_input() -> None:
    assert split_claims("   ") == []


# --- refusal detection -----------------------------------------------------


def test_recognises_refusal_phrasings() -> None:
    assert is_refusal("I don't have enough information to answer that.")
    assert is_refusal("The provided context does not mention revenue.")
    assert is_refusal("The exact percentage is not specified in the context.")


def test_normal_answer_is_not_a_refusal() -> None:
    assert not is_refusal("The refund window is 30 days.")


def test_pure_refusal_yields_no_claims() -> None:
    claims, refused = extract_claims("The context does not specify the exact percentage.")
    assert (claims, refused) == ([], True)


@pytest.mark.parametrize(
    "draft",
    [
        "The context does not state it, but the bonus is 10%.",
        "The context doesn't specify the percentage; however, it is typically 10%.",
        "I don't have enough information. The bonus is 10%.",
    ],
)
def test_claims_hidden_behind_refusal_language_are_extracted(draft: str) -> None:
    """Regression: refusal wording anywhere used to wave the whole draft through."""
    claims, refused = extract_claims(draft)
    assert refused
    assert any("10%" in c for c in claims)


# --- deterministic numeric check -------------------------------------------


def test_number_normalisation() -> None:
    assert numbers_in("1,000 requests") == numbers_in("1000 requests") == {"1000"}
    assert "5" in numbers_in("retried up to five times")
    assert "99.95" in numbers_in("99.95 percent")


def test_unsupported_numbers_are_detected() -> None:
    evidence = "The rate limit is 1000 requests per minute."
    assert unsupported_numbers("The rate limit is 10000 requests per minute.", evidence) == [
        "10000"
    ]
    assert unsupported_numbers("The limit is 1,000 per minute.", evidence) == []


def test_prose_number_words_in_a_claim_are_not_flagged() -> None:
    """'one of', 'first' are prose. Flagging them would make the gate over-refuse."""
    assert unsupported_numbers("One of the options is the first plan.", "Plans exist.") == []


def test_spelled_out_evidence_supports_digit_claims() -> None:
    assert unsupported_numbers("Deliveries are retried 5 times.", "retried up to five times") == []


# --- the gate --------------------------------------------------------------


async def test_fully_grounded_answer_passes() -> None:
    result = await verify_answer(_jev(), CTX, "The refund window is 30 days for enterprise.")
    assert result.passed is True
    assert result.groundedness == 1.0


async def test_fully_ungrounded_answer_is_blocked() -> None:
    result = await verify_answer(_jev(default=0.02), CTX, "Revenue in 2019 was huge.")
    assert result.passed is False
    assert result.groundedness == 0.0


async def test_partially_grounded_answer_is_blocked_at_strict_threshold() -> None:
    answer = (
        "The refund window is 30 days. Requests go through the billing portal. "
        "Enterprise customers qualify. Revenue rose sharply in Zanzibar."
    )
    result = await verify_answer(_jev(("Zanzibar",)), CTX, answer, groundedness_threshold=1.0)
    assert result.passed is False
    assert result.groundedness == 0.75
    assert sum(1 for v in result.verdicts if not v.grounded) == 1


async def test_partially_grounded_answer_passes_at_relaxed_threshold() -> None:
    answer = (
        "The refund window is 30 days. Requests go through the billing portal. "
        "Enterprise customers qualify. Revenue rose sharply in Zanzibar."
    )
    result = await verify_answer(_jev(("Zanzibar",)), CTX, answer, groundedness_threshold=0.7)
    assert result.passed is True


async def test_all_claims_are_verified_in_a_single_request() -> None:
    """Jev evaluates many questions in one call; verification must use that."""
    jev = _jev()
    await verify_answer(jev, CTX, "First claim is here. Second claim is here. Third claim here.")
    assert len(jev.verify_calls) == 1
    assert len(jev.verify_calls[0]) == 3


async def test_holistic_mode_sends_one_combined_claim() -> None:
    jev = _jev()
    result = await verify_answer(jev, CTX, "First claim here. Second claim here.", mode="holistic")
    assert jev.verify_calls == [["First claim here. Second claim here."]]
    assert result.passed is True


async def test_fabricated_number_fails_even_if_the_model_says_supported() -> None:
    """Numbers are checked in code: Jev is documented as weak at numeric comparison."""
    result = await verify_answer(_jev(default=0.99), BONUS, "The bonus is 10% every December.")
    assert result.passed is False
    assert result.verdicts[0].unsupported_numbers == ["10"]
    assert result.verdicts[0].probability == 0.99


async def test_refusal_wrapper_does_not_bypass_the_gate() -> None:
    """Regression for the 'the context does not state it, but it is 10%' bypass."""
    result = await verify_answer(
        _jev(default=0.99), BONUS, "The context does not state it, but the bonus is 10%."
    )
    assert result.passed is False
    assert result.declined is False


async def test_short_fabrication_is_verified_not_skipped() -> None:
    """Regression for the 'It is 10%.' bypass via the minimum-length filter."""
    jev = _jev(default=0.99)
    result = await verify_answer(
        jev, BONUS, "It is 10%.", question="What is the exact bonus percentage?"
    )
    assert result.passed is False
    # The fragment is sent with the question attached so the model can judge it.
    assert "exact bonus percentage" in jev.verify_calls[0][0]


async def test_pure_refusal_is_reported_as_declined() -> None:
    """An honest refusal is not a fabrication, and is flagged as a decline."""
    jev = _jev(default=0.02)
    result = await verify_answer(jev, CTX, "I don't have enough information to answer that.")
    assert result.passed is True
    assert result.declined is True
    assert jev.verify_calls == []


async def test_empty_draft_fails() -> None:
    result = await verify_answer(_jev(), CTX, "   ")
    assert result.passed is False
    assert result.reason == "empty_draft"


async def test_empty_context_always_fails_regardless_of_fail_open() -> None:
    result = await verify_answer(_jev(), [], "Some confident claim about things.", fail_open=True)
    assert result.passed is False
    assert result.reason == "no_context_to_verify_against"


async def test_verifier_outage_fails_closed_by_default() -> None:
    result = await verify_answer(_jev(fail={"verify"}), CTX, "A claim about refunds.")
    assert result.passed is False
    assert result.reason == "verifier_unavailable"


async def test_verifier_outage_can_fail_open_when_configured() -> None:
    result = await verify_answer(_jev(fail={"verify"}), CTX, "A claim.", fail_open=True)
    assert result.passed is True
    assert result.reason == "verifier_unavailable_failed_open"


async def test_malformed_verifier_response_fails_closed() -> None:
    class _Short(ScriptedJev):
        async def verify_claims(self, source, claims):  # type: ignore[no-untyped-def]
            from cribrix.clients.jev import GroundingResult

            return GroundingResult([0.99])  # one probability for two claims

    result = await verify_answer(_Short(), CTX, "First claim here. Second claim here.")
    assert result.passed is False
    assert result.reason == "verifier_unavailable"
