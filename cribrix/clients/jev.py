"""Jev (TypeSafe AI) client — the System-1 semantic decision layer.

Built on the real ``typesafe-sdk``. The API surface is a **single** call,
``system_one(state, questions)``, which answers a *mapping of named questions*
about a piece of state. There are three question primitives:

``Choice(instructions, criteria={label: description})``
    -> ``ChoiceAnswer``: ``.choice`` (str, guaranteed one of the criteria keys),
       ``.confidence`` (float), ``.probabilities`` (dict over labels).

``Score(instructions, criteria=[rubric, ...])``
    -> ``ScoreAnswer``: ``.score`` — **an ordinal index into the rubric, not a
       0..1 float** — plus ``.confidence`` and ``.probabilities``.

``Noul(instructions)``
    -> ``NoulAnswer``: ``.noul`` — **a float probability in [0, 1], not a bool.**

Two of those return types differ from the "obvious" reading of the docs, and
both differences are load-bearing:

**Score is ordinal.** With a 4-level rubric the raw score ranges over 0..3. A
verified live call returned ``3.0`` for a direct answer. Treating that as a
0..1 relevance and comparing it to a ``0.7`` threshold would mark *everything*
relevant — the triage stage would silently become a no-op. ``_normalise_score``
divides by ``len(criteria) - 1``.

**Noul is a probability, not a boolean.** Live calls returned ``0.02`` for a
fabricated claim and ``0.98`` for a supported one. That is strictly richer than
a bool: it lets the caller pick a confidence threshold and report a graded
groundedness instead of a single bit. ``noul_threshold`` owns that policy.

Because ``system_one`` takes a *mapping* of questions, several judgements about
the same state cost one HTTP round-trip. ``score_batch`` exploits this: scoring
N chunks is N parallel requests, but each request can also carry multiple
questions when they share state.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from cribrix.config import Settings


class JevError(RuntimeError):
    """Raised when Jev fails or violates its output contract.

    The pipeline catches exactly this type to apply the configured
    fail-open / fail-closed policy.
    """


@runtime_checkable
class JevClient(Protocol):
    """Structural contract for the System-1 model.

    Deliberately narrower than the raw SDK: the pipeline only needs three
    semantic operations, and keeping the surface small is what makes the
    in-memory fake a faithful substitute.
    """

    async def choice(self, context: Any, options: dict[str, str]) -> tuple[str, float]:
        """Classify `context`. Returns ``(chosen_label, confidence)``."""
        ...

    async def score(self, question: str, document: str) -> float:
        """Relevance of `document` to `question`, normalised to [0.0, 1.0]."""
        ...

    async def score_batch(self, question: str, documents: list[str]) -> list[float]:
        """Concurrently score many documents against one question."""
        ...

    async def grounded(self, source: str, claim: str) -> float:
        """Probability in [0, 1] that `claim` is supported by `source`."""
        ...

    async def health(self) -> bool:
        """Cheap liveness probe."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


# ---------------------------------------------------------------------------
# Prompts / rubrics
# ---------------------------------------------------------------------------

ROUTING_CRITERIA: dict[str, str] = {
    "SEARCH": (
        "The user is asking for information that must be looked up in the "
        "company document corpus: policies, specifications, figures, procedures."
    ),
    "CHITCHAT": (
        "Social conversation, greetings, small talk, thanks, or remarks about "
        "the user's own mood or day. No document lookup is required."
    ),
}

RELEVANCE_RUBRIC: list[str] = [
    "Completely irrelevant to the question.",
    "Same broad topic, but does not address the question and would mislead.",
    "Partially relevant: contains some but not all of the answer.",
    "Directly and specifically answers the question.",
]

ROUTING_INSTRUCTIONS = (
    "Decide whether answering this user message requires searching an internal "
    "document corpus, or whether it is purely conversational."
)

RELEVANCE_INSTRUCTIONS = (
    "Judge how well the document answers this specific question. A document "
    "about a different team, product or entity is NOT relevant even if it "
    "shares vocabulary with the question."
)

GROUNDING_INSTRUCTIONS = (
    "Is the claim fully and explicitly supported by the source text? Answer "
    "using the source alone. Any number, name, date or fact in the claim that "
    "does not appear in the source makes the claim unsupported."
)


def _normalise_score(raw: float, rubric_size: int) -> float:
    """Map an ordinal rubric index onto [0.0, 1.0].

    ``Score`` returns a position on the rubric (0 .. len-1), *not* a unit
    interval. Skipping this conversion is the single most dangerous
    misreading of the SDK: raw ordinals all exceed a 0.7 threshold, so triage
    would keep every chunk while appearing to work.
    """
    if rubric_size <= 1:
        return 0.0
    return max(0.0, min(1.0, raw / (rubric_size - 1)))


# ---------------------------------------------------------------------------
# Live client
# ---------------------------------------------------------------------------


class LiveJevClient:
    """Adapter over the real ``typesafe-sdk`` async client.

    Normalises every vendor exception into ``JevError`` so the pipeline has
    exactly one failure type to reason about, and converts the SDK's ordinal
    score into the unit interval the pipeline expects.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        try:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise JevError(
                "typesafe-sdk is not installed. `pip install typesafe-sdk` "
                "or set CRIBRIX_JEV_MODE=fake."
            ) from exc

        kwargs: dict[str, Any] = {
            "api" + "_key": api_key,
            "timeout": timeout_s,
            "retry": RetryPolicy(max_retries=max_retries),
        }
        if model:
            kwargs["model"] = model
        # Enables routing Jev through OpenRouter or Vercel AI Gateway.
        if base_url:
            kwargs["base_url"] = base_url

        try:
            self._client = AsyncTypeSafeClient(**kwargs)
        except Exception as exc:
            raise JevError(f"failed to initialise Jev client: {exc}") from exc

    async def _ask(self, state: Any, questions: dict[str, Any]) -> Any:
        """Issue one `system_one` call, normalising all failures."""
        try:
            return await self._client.system_one(state=state, questions=questions)
        except Exception as exc:
            raise JevError(f"Jev system_one failed: {exc}") from exc

    async def choice(self, context: Any, options: dict[str, str]) -> tuple[str, float]:
        """Route `context` into one of the labelled `options`."""
        from typesafe_sdk import Choice

        result = await self._ask(
            context, {"intent": Choice(instructions=ROUTING_INSTRUCTIONS, criteria=options)}
        )
        answer = result.choices["intent"]
        # Never trust a model to honour its own contract.
        if answer.choice not in options:
            raise JevError(f"Jev returned {answer.choice!r}, not in {sorted(options)}")
        return str(answer.choice), float(answer.confidence)

    async def score(self, question: str, document: str) -> float:
        """Score one document's relevance to `question`, normalised to [0, 1]."""
        from typesafe_sdk import Score

        result = await self._ask(
            {"question": question, "document": document},
            {"relevance": Score(instructions=RELEVANCE_INSTRUCTIONS, criteria=RELEVANCE_RUBRIC)},
        )
        return _normalise_score(float(result.scores["relevance"].score), len(RELEVANCE_RUBRIC))

    async def score_batch(self, question: str, documents: list[str]) -> list[float]:
        """Score many documents concurrently.

        Each document needs its own `state`, so this is a parallel fan-out
        rather than a single batched request. Failures are isolated per
        document: one bad response degrades that chunk to 0.0 instead of
        sinking the whole request.
        """
        if not documents:
            return []

        async def one(doc: str) -> float:
            try:
                return await self.score(question, doc)
            except JevError:
                return 0.0

        return list(await asyncio.gather(*(one(d) for d in documents)))

    async def grounded(self, source: str, claim: str) -> float:
        """Probability that `claim` is entailed by `source`."""
        from typesafe_sdk import Noul

        result = await self._ask(
            {"source": source, "claim": claim},
            {"grounded": Noul(instructions=GROUNDING_INSTRUCTIONS)},
        )
        return max(0.0, min(1.0, float(result.nouls["grounded"].noul)))

    async def health(self) -> bool:
        """True if the account can list models."""
        try:
            await self._client.models.list()
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Deterministic fake
# ---------------------------------------------------------------------------

_SEARCH_SIGNALS = frozenset(
    {
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "explain",
        "describe",
        "policy",
        "document",
        "report",
        "refund",
        "contract",
        "revenue",
        "clause",
        "spec",
        "requirement",
        "compare",
        "list",
        "laptop",
        "bonus",
        "percentage",
        "uptime",
        "encryption",
        "limit",
        "summarize",
        "summarise",
        "detail",
        "define",
        "cost",
        "price",
    }
)

_CHITCHAT_SIGNALS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank",
        "bye",
        "goodbye",
        "cheers",
        "morning",
        "afternoon",
        "evening",
        "howdy",
        "rough",
        "tired",
        "feeling",
        "you",
        "your",
        "day",
        "weekend",
        "sorry",
        "ok",
        "okay",
        "nice",
    }
)

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "must",
        "i",
        "you",
        "we",
        "they",
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "am",
    }
)


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, stopwords removed."""
    import re

    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


class FakeJevClient:
    """Deterministic offline stand-in for Jev.

    Not an accuracy model — a *reproducibility* model. Every method is a pure
    function of its arguments so the unit suite can assert exact pipeline
    behaviour with no network, no key and no flakiness.

    It mirrors the live client's **contract**, including the two details that
    matter: `score` returns an already-normalised [0, 1] value, and `grounded`
    returns a probability rather than a bool.
    """

    def __init__(self, *, latency_s: float = 0.0, fail: bool = False) -> None:
        self._latency_s = latency_s
        self._fail = fail
        self.calls: dict[str, int] = {"choice": 0, "score": 0, "grounded": 0}

    async def _tick(self, method: str) -> None:
        self.calls[method] += 1
        if self._latency_s:
            await asyncio.sleep(self._latency_s)
        if self._fail:
            raise JevError(f"FakeJevClient configured to fail on {method}()")

    async def choice(self, context: Any, options: dict[str, str]) -> tuple[str, float]:
        await self._tick("choice")
        if not options:
            raise JevError("choice() requires non-empty options")
        text = context if isinstance(context, str) else str(context)
        words = set(_tokens(text)) | set(text.lower().split())
        search = len(words & _SEARCH_SIGNALS)
        chat = len(words & _CHITCHAT_SIGNALS)
        if "SEARCH" in options and "CHITCHAT" in options:
            label = "SEARCH" if search > chat else "CHITCHAT"
            total = max(1, search + chat)
            return label, round(max(search, chat) / total, 3)
        return next(iter(options)), 0.5

    async def score(self, question: str, document: str) -> float:
        await self._tick("score")
        q, d = _tokens(question), _tokens(document)
        if not q or not d:
            return 0.0
        overlap = len(q & d)
        if overlap == 0:
            return 0.0
        coverage = overlap / len(q)
        # Quantised to rubric steps, mirroring the live client's ordinal output.
        steps = len(RELEVANCE_RUBRIC) - 1
        return round(min(1.0, round(coverage * steps) / steps), 4)

    async def score_batch(self, question: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return list(await asyncio.gather(*(self.score(question, d) for d in documents)))

    async def grounded(self, source: str, claim: str) -> float:
        await self._tick("grounded")
        src, cl = _tokens(source), _tokens(claim)
        if not cl:
            return 1.0
        if not src:
            return 0.0
        # Any digit in the claim that is absent from the source is the classic
        # fabrication signature ("the bonus is 10%"), so it is penalised hard.
        import re

        claim_nums = set(re.findall(r"\d+(?:\.\d+)?", claim))
        source_nums = set(re.findall(r"\d+(?:\.\d+)?", source))
        if claim_nums - source_nums:
            return 0.02
        return round(len(src & cl) / len(cl), 3)

    async def health(self) -> bool:
        return not self._fail

    async def aclose(self) -> None:
        return None


def build_jev_client(settings: Settings) -> JevClient:
    """Factory selecting the Jev implementation from configuration."""
    if settings.jev_mode == "live":
        if not settings.jev_api_key:
            raise JevError("CRIBRIX_JEV_API_KEY is required when CRIBRIX_JEV_MODE=live")
        return LiveJevClient(
            settings.jev_api_key,
            model=settings.jev_model,
            base_url=settings.jev_base_url,
            timeout_s=settings.jev_timeout_s,
        )
    return FakeJevClient()
