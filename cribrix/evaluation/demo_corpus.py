"""A ready-to-query demo corpus, so the project is interesting on first run.

Cribrix's whole thesis is *knowing when not to answer*, and that is impossible
to demonstrate against an empty database. This corpus is deliberately built to
exercise every branch of the pipeline:

* **Answerable** questions with exactly one supporting passage.
* **Near-miss distractors** — passages about a *different* team, plan tier or
  product that share vocabulary with the question. These are what separate a
  real triage stage from a cosine-similarity cut-off.
* **Deliberate gaps** — topics the corpus mentions without ever stating the
  specific figure a user will ask for. This is the hallucination trap.

Every chunk is short, plain and human-checkable, so a reviewer can verify the
system's judgements by reading the source rather than trusting a score.
"""

from __future__ import annotations

from dataclasses import dataclass

from cribrix.schemas import Chunk


@dataclass(frozen=True)
class DemoQuestion:
    """A suggested question, with what the pipeline is expected to do."""

    question: str
    expectation: str
    why: str


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

DEMO_CORPUS: list[Chunk] = [
    # -- Billing: tiered policies are a classic distractor pair ---------------
    Chunk(
        id=1,
        document_id="billing-policy",
        content=(
            "Enterprise customers may request a full refund within 30 days of the "
            "invoice date. Refund requests are submitted through the billing portal "
            "and processed within 10 business days."
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
        document_id="billing-policy",
        content=(
            "Invoices are issued on the first business day of each month. "
            "Payment is due within 30 days of issue."
        ),
    ),
    # -- Security -------------------------------------------------------------
    Chunk(
        id=4,
        document_id="security-whitepaper",
        content=(
            "All customer data is encrypted at rest using AES-256 and in transit "
            "using TLS 1.3. Encryption keys are rotated every 90 days."
        ),
    ),
    Chunk(
        id=5,
        document_id="security-whitepaper",
        content=(
            "The platform completed its SOC 2 Type II audit in March 2026 with no "
            "exceptions reported by the auditor."
        ),
    ),
    Chunk(
        id=6,
        document_id="security-whitepaper",
        content=(
            "Production access requires hardware multi-factor authentication. "
            "All administrative actions are logged to an append-only audit trail."
        ),
    ),
    # -- SLA ------------------------------------------------------------------
    Chunk(
        id=7,
        document_id="sla",
        content=(
            "The service level agreement guarantees 99.95 percent monthly uptime "
            "for Enterprise customers. Service credits are issued automatically "
            "when uptime falls below the guarantee."
        ),
    ),
    Chunk(
        id=8,
        document_id="sla",
        content=(
            "Standard plan customers receive a 99.5 percent uptime target. "
            "This target is not backed by service credits."
        ),
    ),
    # -- API ------------------------------------------------------------------
    Chunk(
        id=9,
        document_id="api-reference",
        content=(
            "The rate limit for the public API is 1000 requests per minute per "
            "API key. Exceeding the limit returns HTTP 429 with a Retry-After header."
        ),
    ),
    Chunk(
        id=10,
        document_id="api-reference",
        content=(
            "Webhook deliveries are retried up to five times with exponential "
            "backoff before being marked as failed."
        ),
    ),
    Chunk(
        id=11,
        document_id="api-reference",
        content=(
            "API responses are paginated with a default page size of 50 items and a maximum of 200."
        ),
    ),
    # -- IT assets: the Keyword Mirage, in the live corpus ---------------------
    Chunk(
        id=12,
        document_id="it-assets",
        content="The engineering team uses MacBook Pro M3s.",
    ),
    Chunk(
        id=13,
        document_id="it-assets",
        content="The marketing team uses MacBook Airs.",
    ),
    Chunk(
        id=14,
        document_id="cafeteria-menu",
        content="The cafeteria is now serving mac and cheese on Thursdays.",
    ),
    # -- HR: deliberately incomplete on the bonus figure ----------------------
    Chunk(
        id=15,
        document_id="compensation",
        content=(
            "The company offers a generous annual performance bonus. "
            "The exact percentage is decided by the board every December."
        ),
    ),
    Chunk(
        id=16,
        document_id="compensation",
        content=(
            "Salary reviews take place annually in April. Promotions may be "
            "proposed by a manager at any point in the year."
        ),
    ),
    Chunk(
        id=17,
        document_id="hr-leave",
        content=(
            "Full-time employees accrue 25 days of paid annual leave per year, "
            "plus public holidays. Up to 5 unused days may be carried over."
        ),
    ),
    Chunk(
        id=18,
        document_id="hr-onboarding",
        content=(
            "New workspaces are provisioned within one hour of signup. An "
            "onboarding specialist is assigned to every Enterprise account."
        ),
    ),
    # -- Support --------------------------------------------------------------
    Chunk(
        id=19,
        document_id="support-policy",
        content=(
            "Enterprise support responds to critical incidents within 1 hour, "
            "24 hours a day. Standard support responds within 1 business day."
        ),
    ),
    Chunk(
        id=20,
        document_id="support-policy",
        content=(
            "Support is available in English, German and Japanese. Requests in "
            "other languages are handled on a best-effort basis."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Guided tour
# ---------------------------------------------------------------------------

DEMO_QUESTIONS: list[DemoQuestion] = [
    DemoQuestion(
        question="What is the refund window for enterprise plans?",
        expectation="ANSWERED",
        why="One passage answers it directly; the Standard/Pro tier is filtered out.",
    ),
    DemoQuestion(
        question="What laptop does the engineering team use?",
        expectation="ANSWERED",
        why="Triage drops the marketing laptop and the 'mac and cheese' noise.",
    ),
    DemoQuestion(
        question="How often are encryption keys rotated?",
        expectation="ANSWERED",
        why="A specific figure that is genuinely present in the corpus.",
    ),
    DemoQuestion(
        question="What is the exact percentage of the annual bonus?",
        expectation="ANSWERED, UNGROUNDED or INSUFFICIENT_CONTEXT",
        why=(
            "The corpus confirms a bonus exists but never states a percentage. "
            "Which branch fires depends on the generator: the offline fake is "
            "extractive so it answers honestly (ANSWERED); a weaker live model "
            "invents a figure and is caught by the gate (UNGROUNDED). Both are "
            "correct behaviour - see `make scenarios` for the isolated gate test."
        ),
    ),
    DemoQuestion(
        question="What was the company's total revenue in 2019?",
        expectation="INSUFFICIENT_CONTEXT",
        why="Entirely absent. Naive RAG answers anyway; Cribrix refuses.",
    ),
    DemoQuestion(
        question="What is the penalty for terminating a contract early?",
        expectation="INSUFFICIENT_CONTEXT",
        why="Adjacent to billing, so chunks score moderately — tests calibration.",
    ),
    DemoQuestion(
        question="Hey, I'm having a rough morning, how are you?",
        expectation="CHITCHAT",
        why="Retrieval is skipped entirely; no LLM call is made.",
    ),
]
