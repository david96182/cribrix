"""Evaluation harness: quantifies what the cribrix is actually buying you.

Runs the golden dataset through two configurations and prints a comparison:

* **baseline** — naive RAG. No triage (threshold 0.0), no verification. This is
  what a typical tutorial implementation does.
* **cribrix**    — triage at the configured threshold plus atomic fact-checking.

The headline metric is **refusal accuracy on unanswerable questions**. Any RAG
system can answer answerable questions; only a calibrated one knows when to
stop. A system that answers everything scores 100% on coverage and is useless.

Usage:
    python -m cribrix.evaluation.runner
    python -m cribrix.evaluation.runner --threshold 0.5 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass

from cribrix.clients.jev import FakeJevClient
from cribrix.clients.llm import FakeLLMClient
from cribrix.config import Settings
from cribrix.evaluation.dataset import CORPUS, GOLDEN_CASES, GoldenCase
from cribrix.observability import configure_logging
from cribrix.pipeline.orchestrator import RAGPipeline
from cribrix.pipeline.retrieval import InMemoryRetriever
from cribrix.schemas import AnswerStatus, QueryResponse

REFUSAL_STATUSES = {
    AnswerStatus.INSUFFICIENT_CONTEXT,
    AnswerStatus.NO_DOCUMENTS,
    AnswerStatus.UNGROUNDED,
    AnswerStatus.VERIFIER_UNAVAILABLE,
}


@dataclass
class CaseResult:
    """Outcome of one golden case under one configuration."""

    query: str
    expected: str
    actual: str
    correct: bool
    triage_precision: float | None
    triage_recall: float | None
    groundedness: float | None
    latency_ms: float


@dataclass
class EvalReport:
    """Aggregate metrics for one configuration."""

    config: str
    total: int
    status_accuracy: float
    answerable_accuracy: float
    refusal_accuracy: float
    hallucination_rate: float
    mean_triage_precision: float | None
    mean_triage_recall: float | None
    mean_latency_ms: float
    cases: list[CaseResult]


def _triage_metrics(response: QueryResponse, case: GoldenCase) -> tuple[float | None, float | None]:
    """Precision/recall of the triage stage against ground-truth relevance."""
    if not case.relevant_chunk_ids or response.trace is None:
        return None, None
    kept = {s.chunk.id for s in response.trace.scored_chunks if s.kept}
    if not kept:
        return 0.0, 0.0
    tp = len(kept & case.relevant_chunk_ids)
    precision = tp / len(kept)
    recall = tp / len(case.relevant_chunk_ids)
    return precision, recall


async def run_config(name: str, settings: Settings, *, hallucinate: bool) -> EvalReport:
    """Execute every golden case under one configuration."""
    jev = FakeJevClient()
    llm = FakeLLMClient(hallucinate=hallucinate)
    retriever = InMemoryRetriever(CORPUS)
    pipeline = RAGPipeline(jev=jev, llm=llm, retriever=retriever, settings=settings)

    results: list[CaseResult] = []
    for case in GOLDEN_CASES:
        response = await pipeline.run(case.query, include_trace=True)
        precision, recall = _triage_metrics(response, case)
        results.append(
            CaseResult(
                query=case.query,
                expected=case.expected_status.value,
                actual=response.status.value,
                correct=response.status == case.expected_status,
                triage_precision=precision,
                triage_recall=recall,
                groundedness=response.trace.groundedness if response.trace else None,
                latency_ms=response.trace.total_ms if response.trace else 0.0,
            )
        )

    answerable = [
        r
        for r, c in zip(results, GOLDEN_CASES, strict=True)
        if c.expected_status == AnswerStatus.ANSWERED
    ]
    unanswerable = [
        r
        for r, c in zip(results, GOLDEN_CASES, strict=True)
        if c.expected_status in REFUSAL_STATUSES
    ]

    # A hallucination is the failure mode we actually care about: the system
    # produced a confident answer to a question the corpus cannot support.
    hallucinations = sum(1 for r in unanswerable if r.actual == AnswerStatus.ANSWERED.value)

    precisions = [r.triage_precision for r in results if r.triage_precision is not None]
    recalls = [r.triage_recall for r in results if r.triage_recall is not None]

    def pct(numerator: int, denominator: int) -> float:
        return round(100.0 * numerator / denominator, 1) if denominator else 0.0

    return EvalReport(
        config=name,
        total=len(results),
        status_accuracy=pct(sum(1 for r in results if r.correct), len(results)),
        answerable_accuracy=pct(sum(1 for r in answerable if r.correct), len(answerable)),
        refusal_accuracy=pct(sum(1 for r in unanswerable if r.correct), len(unanswerable)),
        hallucination_rate=pct(hallucinations, len(unanswerable)),
        mean_triage_precision=(round(sum(precisions) / len(precisions), 3) if precisions else None),
        mean_triage_recall=round(sum(recalls) / len(recalls), 3) if recalls else None,
        mean_latency_ms=round(sum(r.latency_ms for r in results) / len(results), 2),
        cases=results,
    )


def _print_table(reports: list[EvalReport]) -> None:
    """Render the comparison as a markdown table, ready to paste into a README."""
    header = f"| {'Metric':<26} | " + " | ".join(f"{r.config:>12}" for r in reports) + " |"
    sep = f"| {'-' * 26} | " + " | ".join("-" * 12 for _ in reports) + " |"
    print("\n" + header)
    print(sep)

    rows: list[tuple[str, list[str]]] = [
        ("Overall status accuracy", [f"{r.status_accuracy}%" for r in reports]),
        ("Answerable correct", [f"{r.answerable_accuracy}%" for r in reports]),
        ("Refusal accuracy", [f"{r.refusal_accuracy}%" for r in reports]),
        ("Hallucination rate", [f"{r.hallucination_rate}%" for r in reports]),
        (
            "Triage precision",
            [
                str(r.mean_triage_precision if r.mean_triage_precision is not None else "n/a")
                for r in reports
            ],
        ),
        (
            "Triage recall",
            [
                str(r.mean_triage_recall if r.mean_triage_recall is not None else "n/a")
                for r in reports
            ],
        ),
        ("Mean latency (ms)", [f"{r.mean_latency_ms}" for r in reports]),
    ]
    for label, values in rows:
        print(f"| {label:<26} | " + " | ".join(f"{v:>12}" for v in values) + " |")

    print("\nPer-case detail (cribrix):")
    for case in reports[-1].cases:
        mark = "PASS" if case.correct else "FAIL"
        print(f"  [{mark}] {case.query[:58]:<58} -> {case.actual}")


async def _main_async(args: argparse.Namespace) -> int:
    baseline = Settings(
        relevance_threshold=0.0,  # keep everything: no cribrix
        min_chunks_required=0,
        retrieval_top_k=5,
        max_chunks_to_llm=5,
        groundedness_threshold=0.0,  # accept anything: no gate
        verification_mode="holistic",
        jev_mode="fake",
        llm_provider="fake",
    )
    tuned = Settings(
        relevance_threshold=args.threshold,
        min_chunks_required=1,
        retrieval_top_k=args.top_k,
        max_chunks_to_llm=8,
        groundedness_threshold=1.0,
        verification_mode="atomic",
        jev_mode="fake",
        llm_provider="fake",
    )

    reports = [
        await run_config("baseline", baseline, hallucinate=args.hallucinate),
        await run_config("cribrix", tuned, hallucinate=args.hallucinate),
    ]

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2))
    else:
        _print_table(reports)

    # Non-zero exit if the tuned config regressed, so CI can gate on it.
    return 0 if reports[-1].status_accuracy >= args.min_accuracy else 1


def main() -> int:
    """CLI entrypoint."""
    # Per-stage logs are noise here; the report is the output.
    configure_logging(level="WARNING", json_output=False)
    parser = argparse.ArgumentParser(description="Cribrix evaluation harness")
    # 0.65, not 0.7: Jev's Score primitive is ordinal, so normalised scores
    # land on rubric steps (0, 0.33, 0.67, 1.0). A 0.7 threshold sits in the
    # dead zone just above the 0.67 step and over-refuses. Use `make eval-sweep`
    # to see the cliff.
    parser.add_argument("--threshold", type=float, default=0.65, help="Relevance threshold.")
    parser.add_argument("--top-k", type=int, default=20, help="Retrieval breadth.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument(
        "--hallucinate",
        action="store_true",
        help="Make the mock generator fabricate a claim, to exercise the gate.",
    )
    parser.add_argument(
        "--min-accuracy", type=float, default=0.0, help="Exit non-zero below this accuracy."
    )
    return asyncio.run(_main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
