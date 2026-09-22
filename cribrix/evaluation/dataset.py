"""Golden dataset for the evaluation harness.

Each case declares the *expected terminal state*, not just an expected string.
That is what lets the harness measure the property that actually matters:
does the system refuse when it should, and answer when it should?

The corpus is small and synthetic on purpose — the harness is demonstrating
the measurement methodology, and a reviewer can read the whole fixture in a
minute and verify the labels themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cribrix.schemas import AnswerStatus, Chunk


@dataclass(frozen=True)
class GoldenCase:
    """One labelled evaluation example."""

    query: str
    expected_status: AnswerStatus
    description: str
    relevant_chunk_ids: set[int] = field(default_factory=set)
    """Ground-truth relevant chunks, used to compute triage precision/recall."""


CORPUS: list[Chunk] = [
    Chunk(
        id=1,
        document_id="billing-policy",
        content=(
            "Enterprise customers may request a full refund within 30 days of the "
            "invoice date. Refund requests must be submitted through the billing "
            "portal and are processed within 10 business days."
        ),
    ),
    Chunk(
        id=2,
        document_id="billing-policy",
        content=(
            "Standard and Pro plans have a 14 day refund window. Partial refunds "
            "are not offered for either plan tier."
        ),
    ),
    Chunk(
        id=3,
        document_id="security-whitepaper",
        content=(
            "All customer data is encrypted at rest using AES-256 and in transit "
            "using TLS 1.3. Encryption keys are rotated every 90 days."
        ),
    ),
    Chunk(
        id=4,
        document_id="security-whitepaper",
        content=(
            "The platform completed its SOC 2 Type II audit in March 2026 with no "
            "exceptions reported by the auditor."
        ),
    ),
    Chunk(
        id=5,
        document_id="sla",
        content=(
            "The service level agreement guarantees 99.95 percent monthly uptime "
            "for Enterprise customers. Service credits are issued automatically "
            "when uptime falls below the guarantee."
        ),
    ),
    Chunk(
        id=6,
        document_id="onboarding-guide",
        content=(
            "New workspaces are provisioned within one hour of signup. An "
            "onboarding specialist is assigned to every Enterprise account."
        ),
    ),
    Chunk(
        id=7,
        document_id="api-reference",
        content=(
            "The rate limit for the public API is 1000 requests per minute per "
            "API key. Exceeding the limit returns HTTP 429 with a Retry-After header."
        ),
    ),
    Chunk(
        id=8,
        document_id="api-reference",
        content=(
            "Webhook deliveries are retried up to five times with exponential "
            "backoff before being marked as failed."
        ),
    ),
]


GOLDEN_CASES: list[GoldenCase] = [
    GoldenCase(
        query="What is the refund window for enterprise plans?",
        expected_status=AnswerStatus.ANSWERED,
        description="Directly answerable from a single high-relevance chunk.",
        relevant_chunk_ids={1},
    ),
    GoldenCase(
        query="How is customer data encrypted at rest and in transit?",
        expected_status=AnswerStatus.ANSWERED,
        description="Directly answerable; tests retrieval of a specific technical fact.",
        relevant_chunk_ids={3},
    ),
    GoldenCase(
        query="What uptime does the SLA guarantee for enterprise customers?",
        expected_status=AnswerStatus.ANSWERED,
        description="Answerable; checks numeric fact grounding.",
        relevant_chunk_ids={5},
    ),
    GoldenCase(
        query="What is the rate limit for the public API per key?",
        expected_status=AnswerStatus.ANSWERED,
        description="Answerable from the API reference.",
        relevant_chunk_ids={7},
    ),
    GoldenCase(
        query="What was the company's total revenue in fiscal year 2019?",
        expected_status=AnswerStatus.INSUFFICIENT_CONTEXT,
        description=(
            "UNANSWERABLE. The corpus contains no financial data. A naive RAG "
            "system will retrieve the nearest chunks anyway and fabricate a "
            "number. This case is the whole point of the project."
        ),
    ),
    GoldenCase(
        query="Which airline alliance does the CEO prefer for long-haul travel?",
        expected_status=AnswerStatus.INSUFFICIENT_CONTEXT,
        description="UNANSWERABLE and fully out of domain.",
    ),
    GoldenCase(
        query="What is the penalty clause for early contract termination?",
        expected_status=AnswerStatus.INSUFFICIENT_CONTEXT,
        description=(
            "UNANSWERABLE but *adjacent* to the corpus — billing chunks will score "
            "moderately. This is the case that tests whether the threshold is "
            "actually calibrated rather than merely present."
        ),
    ),
    GoldenCase(
        query="hey there, good morning!",
        expected_status=AnswerStatus.CHITCHAT,
        description="Conversational turn; retrieval must be skipped entirely.",
    ),
    GoldenCase(
        query="thanks, that was helpful",
        expected_status=AnswerStatus.CHITCHAT,
        description="Conversational closing; must not trigger retrieval.",
    ),
]
