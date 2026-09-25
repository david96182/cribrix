"""Evaluation harness: what the cribrix actually buys you, measured honestly.

Two systems, the same retriever, the same generator, the same questions:

* **naive**    top-k retrieval -> stuff the chunks into a "helpful assistant"
               prompt -> return whatever comes out. No routing, no triage, no
               verification. What RAG tutorials ship.
* **cribrix**  route -> retrieve -> triage -> generate -> verify.

Three backends, and the report always states which one produced the numbers:

``fake``     Deterministic doubles. **Wiring test only** — the fake Jev is a
             token-overlap heuristic, so these numbers say nothing about Jev.
``replay``   Recorded live Jev + LLM responses, committed to the repo. Real
             model behaviour, reproducible offline, and what CI gates on.
``record``   Calls the live APIs and (re)writes the recording.

Scoring
-------
A case is *correct* when:

* answerable    -> status ANSWERED **and** every required fact is in the answer
* unanswerable  -> any refusal status
* chitchat      -> status CHITCHAT

A **hallucination** is an ANSWERED response to an unanswerable question, or an
ANSWERED response to an answerable one that lacks the required facts. Each
rate is reported with a 95% Wilson interval, because with a few dozen cases
the interval is part of the result.

Usage::

    python -m cribrix.evaluation.runner                  # replay (default)
    python -m cribrix.evaluation.runner --backend fake   # wiring check
    python -m cribrix.evaluation.runner --backend record # needs API keys
    python -m cribrix.evaluation.runner --sweep          # threshold sweep, no API calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cribrix.clients.jev import (
    PROMPT_FINGERPRINT,
    FakeJevClient,
    JevClient,
    PassageAssessment,
    build_jev_client,
)
from cribrix.clients.llm import (
    NAIVE_SYSTEM_PROMPT,
    FakeLLMClient,
    LLMClient,
    LLMError,
    build_llm_client,
)
from cribrix.clients.recording import DEFAULT_RECORDINGS_DIR, RecordingMiss, ResponseCache
from cribrix.config import Settings, get_settings
from cribrix.evaluation.dataset import CORPUS, GOLDEN_CASES, GoldenCase, facts_present
from cribrix.observability import configure_logging
from cribrix.pipeline.orchestrator import RAGPipeline
from cribrix.pipeline.retrieval import HashingEmbedder, InMemoryVectorRetriever, Retriever
from cribrix.pipeline.triage import _decide
from cribrix.pipeline.verification import extract_claims
from cribrix.schemas import REFUSAL_STATUSES, AnswerStatus, QueryResponse

DEFAULT_RECORDING = DEFAULT_RECORDINGS_DIR / "golden.json"
NAIVE_TOP_K = 5


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """Outcome of one golden case under one system."""

    query: str
    kind: str
    status: str
    answer: str
    correct: bool
    hallucinated: bool
    kept_ids: list[int] = field(default_factory=list)
    relevant_ids: list[int] = field(default_factory=list)


@dataclass
class Rate:
    """A proportion with its 95% Wilson score interval."""

    hits: int
    total: int

    @property
    def value(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        n = self.total
        if n == 0:
            return 0.0, 0.0
        p = self.value
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, centre - half), min(1.0, centre + half)

    def __str__(self) -> str:
        lo, hi = self.interval()
        return f"{self.hits}/{self.total} ({self.value:.0%}, CI {lo:.0%}-{hi:.0%})"


@dataclass
class EvalReport:
    """Aggregate metrics for one system."""

    system: str
    cases: list[CaseResult]

    def _subset(self, kind: str) -> list[CaseResult]:
        return [c for c in self.cases if c.kind == kind]

    @property
    def accuracy(self) -> Rate:
        return Rate(sum(c.correct for c in self.cases), len(self.cases))

    @property
    def answerable_correct(self) -> Rate:
        sub = self._subset("answerable")
        return Rate(sum(c.correct for c in sub), len(sub))

    @property
    def generation_failures(self) -> Rate:
        """Upstream LLM outages. Counted as misses, but reported so they are visible."""
        return Rate(
            sum(c.status == AnswerStatus.GENERATION_FAILED.value for c in self.cases),
            len(self.cases),
        )

    @property
    def refusal_accuracy(self) -> Rate:
        sub = self._subset("unanswerable")
        return Rate(sum(c.correct for c in sub), len(sub))

    @property
    def chitchat_accuracy(self) -> Rate:
        sub = self._subset("chitchat")
        return Rate(sum(c.correct for c in sub), len(sub))

    @property
    def hallucination_rate(self) -> Rate:
        """Share of ANSWERED responses that were wrong or unsupported."""
        answered = [c for c in self.cases if c.status == AnswerStatus.ANSWERED.value]
        return Rate(sum(c.hallucinated for c in answered), len(answered))

    @property
    def unanswerable_answered(self) -> Rate:
        sub = self._subset("unanswerable")
        return Rate(sum(c.status == AnswerStatus.ANSWERED.value for c in sub), len(sub))

    def triage(self) -> tuple[float | None, float | None]:
        """Micro-averaged precision/recall of kept chunks vs. ground truth."""
        tp = kept = relevant = 0
        for c in self.cases:
            if c.kind != "answerable" or not c.relevant_ids:
                continue
            k, r = set(c.kept_ids), set(c.relevant_ids)
            tp += len(k & r)
            kept += len(k)
            relevant += len(r)
        precision = tp / kept if kept else None
        recall = tp / relevant if relevant else None
        return precision, recall


def score_case(case: GoldenCase, status: AnswerStatus, answer: str) -> tuple[bool, bool]:
    """Return ``(correct, hallucinated)`` for one response."""
    answered = status is AnswerStatus.ANSWERED
    if case.kind == "answerable":
        has_facts = facts_present(answer, case.required_facts)
        return answered and has_facts, answered and not has_facts
    if case.kind == "unanswerable":
        return status in REFUSAL_STATUSES, answered
    return status is AnswerStatus.CHITCHAT, False


# ---------------------------------------------------------------------------
# Systems under test
# ---------------------------------------------------------------------------


async def run_naive(
    llm: LLMClient, retriever: Retriever, case: GoldenCase, top_k: int = NAIVE_TOP_K
) -> tuple[AnswerStatus, str, list[int]]:
    """Top-k retrieval straight into a generic prompt, no gates.

    To be fair to the baseline, a draft that is *purely* an admission that the
    context lacks the answer is credited as DECLINED — the same rule Cribrix's
    verifier applies. Modern models often refuse unprompted, and counting that
    as a hallucination would flatter Cribrix.
    """
    chunks = await retriever.retrieve(case.query, top_k=top_k)
    ids = [c.id for c in chunks]
    try:
        answer = await llm.generate(case.query, chunks, system_prompt=NAIVE_SYSTEM_PROMPT)
    except LLMError:
        return AnswerStatus.GENERATION_FAILED, "", ids
    claims, refused = extract_claims(answer)
    if refused and not claims:
        return AnswerStatus.DECLINED, answer, ids
    return AnswerStatus.ANSWERED, answer, ids


PIPELINE_KNOBS = (
    "relevance_threshold",
    "evidence_threshold",
    "injection_max",
    "chitchat_min_confidence",
    "min_chunks_required",
    "max_chunks_to_llm",
    "verification_mode",
    "groundedness_threshold",
    "noul_threshold",
    "fail_open_on_verifier_error",
    "llm_temperature",
    "llm_max_tokens",
)


def cribrix_settings(base: Settings, **overrides: Any) -> Settings:
    """The evaluated configuration: the *code* defaults plus CLI overrides.

    Credentials and provider come from the environment, but every pipeline
    threshold is pinned to the shipped defaults, so a local `.env` cannot make
    the published numbers irreproducible.
    """
    pinned = {name: Settings.model_fields[name].default for name in PIPELINE_KNOBS}
    return base.model_copy(update={**pinned, **overrides})


async def evaluate(
    jev: JevClient,
    llm: LLMClient,
    settings: Settings,
    cases: list[GoldenCase] | None = None,
    *,
    concurrency: int = 4,
) -> tuple[EvalReport, EvalReport, list[QueryResponse]]:
    """Run every case through both systems."""
    cases = cases or GOLDEN_CASES
    embedder = HashingEmbedder(dimension=settings.embedding_dim)
    retriever = InMemoryVectorRetriever(CORPUS, embedder)
    pipeline = RAGPipeline(jev=jev, llm=llm, retriever=retriever, settings=settings)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: GoldenCase) -> tuple[CaseResult, CaseResult, QueryResponse]:
        async with semaphore:
            n_status, n_answer, n_ids = await run_naive(llm, retriever, case)
            response = await pipeline.run(case.query, include_trace=True)
        n_ok, n_hall = score_case(case, n_status, n_answer)
        c_ok, c_hall = score_case(case, response.status, response.answer)
        kept = (
            [s.chunk.id for s in response.trace.scored_chunks if s.kept] if response.trace else []
        )
        relevant = sorted(case.relevant_ids)
        naive_result = CaseResult(
            case.query, case.kind, n_status.value, n_answer, n_ok, n_hall, n_ids, relevant
        )
        guarded_result = CaseResult(
            case.query, case.kind, response.status.value, response.answer, c_ok, c_hall,
            kept, relevant,
        )  # fmt: skip
        return naive_result, guarded_result, response

    rows = await asyncio.gather(*(one(c) for c in cases))
    return (
        EvalReport("naive", [r[0] for r in rows]),
        EvalReport("cribrix", [r[1] for r in rows]),
        [r[2] for r in rows],
    )


# ---------------------------------------------------------------------------
# Threshold sweep (re-scores stored signals; no API calls)
# ---------------------------------------------------------------------------


def sweep(
    responses: list[QueryResponse],
    cases: list[GoldenCase],
    relevance_grid: list[float],
    evidence_grid: list[float],
    injection_max: float,
) -> list[tuple[float, float, int, int, int, int]]:
    """Re-run the triage decision rule over recorded signals.

    Triage decisions are made in code from stored probabilities, so any
    threshold combination can be evaluated without another API call. This
    measures the *triage gate only*: for each setting it reports how many
    answerable questions keep at least one truly relevant chunk, and how many
    unanswerable questions correctly keep nothing.

    Returns rows of ``(relevance, evidence, answerable_ok, answerable_total,
    refused_ok, refused_total)``.
    """
    rows = []
    for rel in relevance_grid:
        for ev in evidence_grid:
            a_ok = a_n = u_ok = u_n = 0
            for case, response in zip(cases, responses, strict=True):
                if case.kind == "chitchat" or response.trace is None:
                    continue
                if not response.trace.scored_chunks:
                    continue
                kept = {
                    s.chunk.id
                    for s in response.trace.scored_chunks
                    if _decide(
                        PassageAssessment(s.relevance, 0.0, s.answers, s.injection),
                        threshold=rel,
                        evidence_threshold=ev,
                        injection_max=injection_max,
                    )
                    is None
                }
                if case.kind == "answerable":
                    a_n += 1
                    a_ok += bool(kept & case.relevant_ids)
                else:
                    u_n += 1
                    u_ok += not kept
            rows.append((rel, ev, a_ok, a_n, u_ok, u_n))
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_BACKEND_BANNER = {
    "fake": (
        "BACKEND: fake — deterministic doubles. This is a WIRING TEST. The fake Jev is a\n"
        "token-overlap heuristic; these numbers are not evidence about Jev's accuracy."
    ),
    "replay": "BACKEND: replay — recorded live Jev + LLM responses (reproducible, offline).",
    "record": "BACKEND: record — live Jev + LLM APIs, responses written to the recording.",
}


def print_report(
    naive: EvalReport, guarded: EvalReport, backend: str, meta: dict[str, Any]
) -> None:
    """Render the comparison as a markdown table, ready to paste into a README."""
    print("\n" + _BACKEND_BANNER[backend])
    if meta:
        details = ", ".join(f"{k}={meta[k]}" for k in sorted(meta) if not k.startswith("_"))
        print(f"  {details}")

    rows = [
        ("Overall correct", naive.accuracy, guarded.accuracy),
        ("Answerable: correct facts", naive.answerable_correct, guarded.answerable_correct),
        ("Unanswerable: refused", naive.refusal_accuracy, guarded.refusal_accuracy),
        ("Chitchat: handled", naive.chitchat_accuracy, guarded.chitchat_accuracy),
        ("Unanswerable: answered anyway", naive.unanswerable_answered,
         guarded.unanswerable_answered),
        ("Answers that were wrong/unsupported", naive.hallucination_rate,
         guarded.hallucination_rate),
        ("LLM upstream failures (counted as misses)", naive.generation_failures,
         guarded.generation_failures),
    ]  # fmt: skip
    width = 42
    print(f"\n| {'Metric':<{width}} | {'naive':<24} | {'cribrix':<24} |")
    print(f"|{'-' * (width + 2)}|{'-' * 26}|{'-' * 26}|")
    for label, a, b in rows:
        print(f"| {label:<{width}} | {a!s:<24} | {b!s:<24} |")
    np_, nr = naive.triage()
    cp, cr = guarded.triage()

    def f(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.2f}"

    print(f"| {'Context precision (answerable)':<{width}} | {f(np_):<24} | {f(cp):<24} |")
    print(f"| {'Context recall (answerable)':<{width}} | {f(nr):<24} | {f(cr):<24} |")

    print("\nPer-case detail (cribrix):")
    for c in guarded.cases:
        mark = "PASS" if c.correct else ("HALL" if c.hallucinated else "FAIL")
        print(f"  [{mark}] {c.kind[:6]:<6} {c.query[:60]:<60} -> {c.status}")


def print_sweep(rows: list[tuple[float, float, int, int, int, int]]) -> None:
    print("\nTriage-gate sweep over recorded signals (no API calls):")
    print("  relevance  evidence   answerable kept a relevant chunk   unanswerable kept nothing")
    for rel, ev, a_ok, a_n, u_ok, u_n in rows:
        print(f"  {rel:>9.2f}  {ev:>8.2f}   {a_ok:>3}/{a_n:<3} ({a_ok / max(a_n, 1):>4.0%})"
              f"{'':>16}{u_ok:>3}/{u_n:<3} ({u_ok / max(u_n, 1):>4.0%})")  # fmt: skip


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _clients(
    backend: str, settings: Settings, recording: Path, *, retry_errors: bool = False
) -> tuple[JevClient, LLMClient, ResponseCache | None]:
    if backend == "fake":
        return FakeJevClient(), FakeLLMClient(), None
    if backend == "replay":
        cache = ResponseCache(recording, mode="replay")
        return (
            build_jev_client(settings, cache=cache),
            build_llm_client(settings, cache=cache),
            cache,
        )
    live = settings.model_copy(update={"jev_mode": "live"})
    if live.llm_provider == "fake":
        raise SystemExit("record needs a real LLM: set CRIBRIX_LLM_PROVIDER in .env")
    cache = ResponseCache(recording, mode="record")
    cache.update_meta(recorded_at=datetime.now(UTC).date().isoformat())
    llm = build_llm_client(live, cache=cache, retry_errors=retry_errors)
    return build_jev_client(live, cache=cache), llm, cache


async def _main_async(args: argparse.Namespace) -> int:
    base = get_settings()
    overrides: dict[str, Any] = {}
    if args.threshold is not None:
        overrides["relevance_threshold"] = args.threshold
    if args.evidence is not None:
        overrides["evidence_threshold"] = args.evidence
    # Recordings are keyed on the exact chunks retrieved, so the evaluation
    # pins its own retrieval parameters instead of inheriting them from .env.
    overrides.update(retrieval_top_k=args.top_k, embedding_dim=384, jev_max_concurrency=4)
    settings = cribrix_settings(base, **overrides)

    jev, llm, cache = _clients(
        args.backend, settings, Path(args.recording), retry_errors=args.retry_errors
    )
    try:
        naive, guarded, responses = await evaluate(jev, llm, settings)
    except RecordingMiss as exc:
        print(f"\nRecording miss: {exc}", file=sys.stderr)
        print(
            "The recording does not cover this configuration. Record it with "
            "`make eval-record` or use `--backend fake`.",
            file=sys.stderr,
        )
        return 2
    finally:
        await jev.aclose()
        await llm.aclose()

    meta = dict(cache.meta) if cache else {}
    meta.pop("jev_prompts", None)
    meta.update(
        relevance_threshold=settings.relevance_threshold,
        evidence_threshold=settings.evidence_threshold,
        cases=len(GOLDEN_CASES),
    )

    if args.json:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "meta": meta,
                    "prompts": PROMPT_FINGERPRINT,
                    "naive": [asdict(c) for c in naive.cases],
                    "cribrix": [asdict(c) for c in guarded.cases],
                },
                indent=2,
            )
        )
    else:
        print_report(naive, guarded, args.backend, meta)

    if args.sweep:
        grid = [0.0, 0.2, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.9]
        ev_grid = [0.0, 0.3, 0.5, 0.7]
        print_sweep(sweep(responses, GOLDEN_CASES, grid, ev_grid, settings.injection_max))

    # Non-zero exit if the guarded system regressed, so CI can gate on it.
    ok = guarded.accuracy.value * 100 >= args.min_accuracy
    ok = ok and guarded.hallucination_rate.value * 100 <= args.max_hallucination
    return 0 if ok else 1


def main() -> int:
    """CLI entrypoint."""
    # Per-stage logs (including expected generation failures) are noise here.
    configure_logging(level="CRITICAL", json_output=False)
    parser = argparse.ArgumentParser(description="Cribrix evaluation harness")
    parser.add_argument("--backend", choices=["replay", "fake", "record"], default="replay")
    parser.add_argument("--recording", default=str(DEFAULT_RECORDING))
    parser.add_argument("--threshold", type=float, default=None, help="Relevance threshold.")
    parser.add_argument("--evidence", type=float, default=None, help="Evidence threshold.")
    parser.add_argument("--top-k", type=int, default=20, help="Retrieval breadth.")
    parser.add_argument("--sweep", action="store_true", help="Also sweep triage thresholds.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --backend record: re-attempt generations that previously failed upstream.",
    )
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--max-hallucination", type=float, default=100.0)
    return asyncio.run(_main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
