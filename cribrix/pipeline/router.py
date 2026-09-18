"""Stage 1 — Intent routing.

Decides whether a turn needs the corpus (`SEARCH`) or is conversational
(`CHITCHAT`). Getting this wrong is asymmetric:

* CHITCHAT misrouted as SEARCH — wasteful, but harmless; triage discards the
  junk and the pipeline refuses cleanly.
* SEARCH misrouted as CHITCHAT — the user's real question is silently answered
  from the model's parametric memory, with no retrieval and no fact-check.
  This is a hallucination delivered with full confidence.

The costs are not symmetric, so neither is the policy: **on any routing
failure we default to SEARCH.** An unnecessary retrieval is cheaper than an
unverified answer.
"""

from __future__ import annotations

from cribrix.clients.jev import ROUTING_CRITERIA, JevClient, JevError
from cribrix.observability import get_logger
from cribrix.schemas import Intent

logger = get_logger(__name__)


async def route_intent(jev: JevClient, query: str) -> tuple[Intent, float]:
    """Classify `query` as SEARCH or CHITCHAT.

    Args:
        jev: System-1 client.
        query: Raw user input.

    Returns:
        ``(intent, confidence)``. Defaults to ``(Intent.SEARCH, 0.0)`` when the
        classifier errors or returns an unrecognised label — see the module
        docstring for why the failure direction is not symmetric.
    """
    try:
        label, confidence = await jev.choice(context=query, options=ROUTING_CRITERIA)
    except JevError:
        logger.warning("router.failed_defaulting_to_search", exc_info=True)
        return Intent.SEARCH, 0.0

    try:
        intent = Intent(label.strip().upper())
    except ValueError:
        logger.warning("router.unknown_label_defaulting_to_search", label=label)
        return Intent.SEARCH, 0.0

    logger.info("router.decided", intent=intent.value, confidence=round(confidence, 3))
    return intent, confidence
