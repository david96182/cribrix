"""Stage 1 — Intent routing.

Decides whether a turn needs the corpus (`SEARCH`) or is conversational
(`CHITCHAT`). Getting this wrong is asymmetric:

* CHITCHAT misrouted as SEARCH — wasteful, but harmless; triage discards the
  junk and the pipeline refuses cleanly.
* SEARCH misrouted as CHITCHAT — the user's real question is answered with a
  canned greeting and no retrieval. The user loses an answer they were owed.

The costs are not symmetric, so neither is the policy. This is the
*confidence-gated routing* pattern from the TypeSafe docs: the Choice says
*what*, its confidence says *whether to act*. We only take the cheap
CHITCHAT exit when the model is confident; anything uncertain — or any error —
falls through to SEARCH.
"""

from __future__ import annotations

from cribrix.clients.jev import ROUTING_CRITERIA, JevClient, JevError
from cribrix.observability import get_logger
from cribrix.schemas import Intent

logger = get_logger(__name__)


async def route_intent(
    jev: JevClient, query: str, *, chitchat_min_confidence: float = 0.8
) -> tuple[Intent, float]:
    """Classify `query` as SEARCH or CHITCHAT.

    Args:
        jev: System-1 client.
        query: Raw user input.
        chitchat_min_confidence: Minimum Choice confidence required to skip
            retrieval. Below it, the query is searched anyway.

    Returns:
        ``(intent, confidence)``. Defaults to ``(Intent.SEARCH, 0.0)`` when the
        classifier errors or returns an unrecognised label.
    """
    try:
        decision = await jev.route(query, ROUTING_CRITERIA)
    except JevError:
        logger.warning("router.failed_defaulting_to_search", exc_info=True)
        return Intent.SEARCH, 0.0

    try:
        intent = Intent(decision.label.strip().upper())
    except ValueError:
        logger.warning("router.unknown_label_defaulting_to_search", label=decision.label)
        return Intent.SEARCH, 0.0

    if intent is Intent.CHITCHAT and decision.confidence < chitchat_min_confidence:
        logger.info(
            "router.low_confidence_chitchat_searched_instead",
            confidence=round(decision.confidence, 3),
            threshold=chitchat_min_confidence,
        )
        return Intent.SEARCH, decision.confidence

    logger.info("router.decided", intent=intent.value, confidence=round(decision.confidence, 3))
    return intent, decision.confidence
