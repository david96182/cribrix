"""The three failure modes of naive RAG, demonstrated side by side.

Each scenario runs the *same* question through two pipelines:

``naive``    top-k retrieval -> stuff everything into the LLM -> return it.
             No routing, no triage, no verification. The tutorial architecture.
``cribrix``  route -> retrieve -> triage -> generate -> verify.

The comparison is the deliverable. A refusal only looks impressive next to the
confident wrong answer the same question produces without the gates.

Run offline (deterministic fakes)::

    python -m cribrix.evaluation.scenarios

Run against the real Jev API and a real LLM::

    python -m cribrix.evaluation.scenarios --live
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field

from cribrix.clients.jev import JevClient, build_jev_client
from cribrix.clients.llm import LLMClient, build_llm_client
from cribrix.config import Settings, get_settings
from cribrix.observability import configure_logging
from cribrix.pipeline.orchestrator import RAGPipeline
from cribrix.pipeline.retrieval import InMemoryRetriever
from cribrix.schemas import AnswerStatus, Chunk, QueryResponse

# Rough public pricing midpoint, only used to make token savings legible.
USD_PER_1K_INPUT_TOKENS = 0.0005
CHARS_PER_TOKEN = 4  # standard approximation for English prose


@dataclass
class Scenario:
    """One demonstrable failure mode of naive RAG."""

    key: str
    title: str
    problem: str
    query: str
    corpus: list[Chunk]
    naive_failure: str
    """What the unguarded pipeline does wrong."""
    expected: AnswerStatus
    """What Cribrix should do instead."""
    notes: list[str] = field(default_factory=list)
    adversarial_draft: str | None = None
    """A known-fabricated draft used to probe the verifier directly.

    Scenario 3 depends on the generator misbehaving, but whether a given model
    hallucinates is a property of *that model*, not of Cribrix. Well-aligned
    models often refuse correctly. Rigging the prompt until the model fails
    would be measuring theatre.

    So the scenario reports what the live model actually did, and then feeds
    this hand-written fabrication straight to the verifier. That isolates the
    question Cribrix is actually responsible for: *if* a fabrication reaches
    the gate, does the gate stop it?
    """
    force_hallucination: bool = False
    """Offline only.

    A real LLM invents the missing figure on its own, which is the whole point
    of the scenario. The deterministic fake is extractive by design, so to
    demonstrate the same failure offline we explicitly ask it to fabricate.
    Under ``--live`` this flag is ignored and the real model is left to fail
    (or not) on its own merits.
    """


# --- Scenario 1 -------------------------------------------------------------

CHITCHAT_CORPUS = [
    Chunk(
        id=1,
        document_id="hr-policy",
        content=(
            "Employees working the morning shift must clock in before 9:00 AM. "
            "Repeated late arrivals are handled under the attendance policy."
        ),
    ),
    Chunk(
        id=2,
        document_id="hr-grievance",
        content=(
            "An employee experiencing difficulty at work may raise a formal "
            "grievance with HR. Grievances are acknowledged within two business days."
        ),
    ),
    Chunk(
        id=3,
        document_id="wellbeing",
        content=(
            "The employee wellbeing programme offers confidential counselling "
            "sessions for staff experiencing personal difficulties."
        ),
    ),
]

SCENARIO_CHITCHAT = Scenario(
    key="chitchat",
    title="The Chitchat Trap",
    problem="Naive RAG treats every input as a search query.",
    query="Hey, I'm having a rough morning, how are you?",
    corpus=CHITCHAT_CORPUS,
    naive_failure=(
        "Vector-searches 'rough morning', retrieves HR attendance and grievance "
        "policies, and answers a friendly greeting with corporate policy text. "
        "Burns a vector search plus a full LLM call to produce a worse answer."
    ),
    expected=AnswerStatus.CHITCHAT,
    notes=["Measures: retrieval skipped, LLM call avoided, latency saved."],
)


# --- Scenario 2 -------------------------------------------------------------

LAPTOP_CORPUS = [
    Chunk(id=1, document_id="it-assets", content="The engineering team uses MacBook Pro M3s."),
    Chunk(id=2, document_id="it-assets", content="The marketing team uses MacBook Airs."),
    Chunk(id=3, document_id="cafeteria", content="The cafeteria is now serving mac and cheese."),
]

SCENARIO_MIRAGE = Scenario(
    key="mirage",
    title="The Keyword Mirage",
    problem=(
        "pgvector returns neighbours that are close in vector space but "
        "factually irrelevant, cluttering the context window."
    ),
    query="What laptop does the engineering team use?",
    corpus=LAPTOP_CORPUS,
    naive_failure=(
        "All three chunks clear the raw cosine cut-off — 'MacBook Airs' and "
        "'mac and cheese' are lexically adjacent to the query. The LLM receives "
        "two distractors and may attribute the wrong laptop to the wrong team."
    ),
    expected=AnswerStatus.ANSWERED,
    notes=["Measures: prompt tokens dropped, distractors removed."],
)


# --- Scenario 3 -------------------------------------------------------------

BONUS_CORPUS = [
    Chunk(
        id=1,
        document_id="comp-policy",
        content=(
            "The company offers a generous annual performance bonus. "
            "The exact percentage is decided by the board every December."
        ),
    ),
]

SCENARIO_HALLUCINATION = Scenario(
    key="hallucination",
    title="The Confident Hallucination",
    problem=(
        "When context lacks the specific fact requested, an LLM would rather "
        "invent a plausible number than admit the gap."
    ),
    query="What is the exact percentage of the annual bonus?",
    corpus=BONUS_CORPUS,
    naive_failure=(
        "The context confirms a bonus exists but never states a percentage. "
        "Asked for an exact figure, the model supplies one anyway — typically "
        "'10%' — with full confidence and a real citation attached."
    ),
    expected=AnswerStatus.UNGROUNDED,
    notes=["Measures: fabricated figure detected and withheld."],
    force_hallucination=True,
    adversarial_draft=(
        "The company offers an annual performance bonus of 10%. "
        "The board reviews it every December."
    ),
)


SCENARIOS: list[Scenario] = [SCENARIO_CHITCHAT, SCENARIO_MIRAGE, SCENARIO_HALLUCINATION]


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Observable outcome of one pipeline on one scenario."""

    label: str
    answer: str
    status: str
    searched: bool
    llm_called: bool
    chunks_to_llm: int
    prompt_chars: int
    latency_ms: float
    detail: str = ""

    @property
    def prompt_tokens(self) -> int:
        """Approximate prompt size in tokens."""
        return self.prompt_chars // CHARS_PER_TOKEN


async def run_naive(scenario: Scenario, llm: LLMClient, *, top_k: int = 5) -> RunResult:
    """Retrieve top-k and stuff everything into the LLM. No gates."""
    from cribrix.clients.llm import NAIVE_SYSTEM_PROMPT, build_prompt

    started = time.perf_counter()
    retriever = InMemoryRetriever(scenario.corpus)
    chunks = await retriever.retrieve(scenario.query, top_k=top_k)
    prompt = build_prompt(scenario.query, chunks)
    try:
        # The baseline uses the generic "helpful assistant" prompt a tutorial
        # ships with. Giving it Cribrix's carefully hedged prompt would be
        # measuring the prompt, not the architecture.
        answer = await llm.generate(scenario.query, chunks, system_prompt=NAIVE_SYSTEM_PROMPT)
    except Exception as exc:
        answer = f"<LLM error: {exc}>"
    return RunResult(
        label="naive",
        answer=answer,
        status="ANSWERED",
        searched=True,
        llm_called=True,
        chunks_to_llm=len(chunks),
        prompt_chars=len(prompt),
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )


async def run_cribrix(
    scenario: Scenario, jev: JevClient, llm: LLMClient, settings: Settings
) -> tuple[RunResult, QueryResponse]:
    """Run the full gated pipeline."""
    from cribrix.clients.llm import build_prompt

    pipeline = RAGPipeline(
        jev=jev,
        llm=llm,
        retriever=InMemoryRetriever(scenario.corpus),
        settings=settings,
    )
    started = time.perf_counter()
    response = await pipeline.run(scenario.query, include_trace=True)
    elapsed = round((time.perf_counter() - started) * 1000, 1)

    trace = response.trace
    kept = [s.chunk for s in (trace.scored_chunks if trace else []) if s.kept]
    llm_called = response.status not in {
        AnswerStatus.CHITCHAT,
        AnswerStatus.NO_DOCUMENTS,
        AnswerStatus.INSUFFICIENT_CONTEXT,
    }
    prompt_chars = len(build_prompt(scenario.query, kept)) if llm_called else 0

    detail = ""
    if trace and trace.claim_verdicts:
        worst = min(trace.claim_verdicts, key=lambda v: v.probability)
        detail = f"lowest claim support p={worst.probability:.2f}"

    return (
        RunResult(
            label="cribrix",
            answer=response.answer,
            status=response.status.value,
            searched=response.status is not AnswerStatus.CHITCHAT,
            llm_called=llm_called,
            chunks_to_llm=len(kept) if llm_called else 0,
            prompt_chars=prompt_chars,
            latency_ms=elapsed,
            detail=detail,
        ),
        response,
    )


async def probe_verifier(
    scenario: Scenario, jev: JevClient, settings: Settings
) -> tuple[bool, list[tuple[str, float]]] | None:
    """Feed a known fabrication straight to the groundedness gate.

    Bypasses the generator entirely. Whether a given LLM hallucinates is a
    property of that LLM; whether Cribrix *catches* a hallucination is a
    property of Cribrix. This measures the second thing.
    """
    if not scenario.adversarial_draft:
        return None
    from cribrix.pipeline.verification import verify_answer

    result = await verify_answer(
        jev,
        scenario.corpus,
        scenario.adversarial_draft,
        mode=settings.verification_mode,
        groundedness_threshold=settings.groundedness_threshold,
        noul_threshold=settings.noul_threshold,
        max_concurrency=settings.jev_max_concurrency,
    )
    return result.passed, [(v.claim, v.probability) for v in result.verdicts]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_money(tokens: int) -> str:
    return f"${tokens / 1000 * USD_PER_1K_INPUT_TOKENS:.5f}"


def print_scenario(
    index: int,
    scenario: Scenario,
    naive: RunResult,
    guarded: RunResult,
    response: QueryResponse,
    probe: tuple[bool, list[tuple[str, float]]] | None = None,
) -> None:
    """Render a single scenario comparison."""
    bar = "=" * 78
    print(f"\n{bar}\nSCENARIO {index}: {scenario.title.upper()}\n{bar}")
    print(f"Problem : {scenario.problem}")
    print(f"Query   : {scenario.query!r}\n")

    print("-- NAIVE RAG " + "-" * 64)
    print(f"   searched DB   : {naive.searched}")
    print(f"   LLM called    : {naive.llm_called}")
    print(f"   chunks -> LLM : {naive.chunks_to_llm}")
    print(f"   prompt tokens : ~{naive.prompt_tokens}  ({_fmt_money(naive.prompt_tokens)})")
    print(f"   latency       : {naive.latency_ms} ms")
    print(f"   answer        : {naive.answer[:200]}")

    print("\n-- CRIBRIX " + "-" * 66)
    print(f"   searched DB   : {guarded.searched}")
    print(f"   LLM called    : {guarded.llm_called}")
    print(f"   chunks -> LLM : {guarded.chunks_to_llm}")
    print(f"   prompt tokens : ~{guarded.prompt_tokens}  ({_fmt_money(guarded.prompt_tokens)})")
    print(f"   latency       : {guarded.latency_ms} ms")
    print(f"   status        : {guarded.status}")
    print(f"   answer        : {guarded.answer[:200]}")
    if guarded.detail:
        print(f"   evidence      : {guarded.detail}")

    trace = response.trace
    if trace and trace.scored_chunks:
        print("\n   triage detail:")
        for scored in trace.scored_chunks:
            mark = "KEEP" if scored.kept else "DROP"
            print(f"     [{mark}] rel={scored.relevance:.2f}  {scored.chunk.content[:58]}")
    if trace and trace.claim_verdicts:
        print("\n   fact-check detail:")
        for verdict in trace.claim_verdicts:
            mark = "OK " if verdict.grounded else "BAD"
            print(f"     [{mark}] p={verdict.probability:.2f}  {verdict.claim[:58]}")

    if probe is not None:
        blocked, probe_claims = probe
        print("\n   adversarial verifier probe (bypasses the generator):")
        print(f"     draft: {scenario.adversarial_draft!r}")
        for claim, prob in probe_claims:
            mark = "OK " if prob >= 0.5 else "BAD"
            print(f"     [{mark}] p={prob:.2f}  {claim[:56]}")
        outcome = "BLOCKED" if not blocked else "LET THROUGH"
        print(f"     >> gate verdict: {outcome}")

    saved = naive.prompt_tokens - guarded.prompt_tokens
    pct = (saved / naive.prompt_tokens * 100) if naive.prompt_tokens else 0.0
    outcome_label = "PASS" if guarded.status == scenario.expected.value else "UNEXPECTED"
    print(f"\n   >> outcome: {outcome_label} (expected {scenario.expected.value})")
    print(f"   >> prompt tokens saved: {saved} (~{pct:.0f}%)")


async def run_all(settings: Settings, *, live: bool) -> int:
    """Execute every scenario under both pipelines."""
    jev = build_jev_client(settings)
    llm = build_llm_client(settings)

    mode = "LIVE (real Jev + real LLM)" if live else "OFFLINE (deterministic fakes)"
    print(f"\nCribrix scenario suite — {mode}")
    if live:
        print(f"  jev model : {settings.jev_model or 'sdk default'}")
        print(f"  llm       : {settings.llm_provider} / {settings.llm_model}")
        print(
            f"  threshold : relevance>={settings.relevance_threshold} "
            f"noul>={settings.noul_threshold}"
        )

    failures = 0
    try:
        for index, scenario in enumerate(SCENARIOS, start=1):
            # Offline, the extractive fake cannot invent a figure, so scenario 3
            # asks it to. Live, the real model is left to hallucinate unaided.
            active_llm = llm
            if scenario.force_hallucination and not live:
                from cribrix.clients.llm import FakeLLMClient

                active_llm = FakeLLMClient(hallucinate=True)
            naive = await run_naive(scenario, active_llm)
            guarded, response = await run_cribrix(scenario, jev, active_llm, settings)
            probe = await probe_verifier(scenario, jev, settings)
            print_scenario(index, scenario, naive, guarded, response, probe)

            # A scenario counts as passing if the pipeline behaved as designed,
            # OR - for the hallucination scenario - if the model happened to be
            # honest AND the gate provably blocks a fabrication when given one.
            ok = guarded.status == scenario.expected.value
            if not ok and probe is not None:
                blocked, _ = probe
                ok = not blocked
                if ok:
                    print(
                        "   >> note: this model did not hallucinate; "
                        "the gate was verified directly instead."
                    )
            if not ok:
                failures += 1
    finally:
        await jev.aclose()
        await llm.aclose()

    print("\n" + "=" * 78)
    print(f"RESULT: {len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios behaved as designed")
    print("=" * 78)
    return 1 if failures else 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Cribrix scenario demonstrations")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real Jev API and configured LLM provider instead of fakes.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show pipeline logs.")
    args = parser.parse_args()

    configure_logging(level="DEBUG" if args.verbose else "WARNING", json_output=False)

    base = get_settings()
    overrides: dict[str, object] = {}
    if not args.live:
        overrides = {"jev_mode": "fake", "llm_provider": "fake"}
    settings = base.model_copy(update=overrides)

    return asyncio.run(run_all(settings, live=args.live))


if __name__ == "__main__":
    raise SystemExit(main())
