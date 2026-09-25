"""Golden dataset for the evaluation harness.

Every case is labelled with:

* ``kind``            ``answerable`` / ``unanswerable`` / ``chitchat``.
* ``relevant_ids``    chunks that genuinely contain the answer (triage ground truth).
* ``required_facts``  strings a *correct* answer must contain. Scoring on
                      facts, not just status, means an ANSWERED response that
                      states the wrong figure counts as a failure — which is
                      the "faithfully grounded in the wrong passage" case.

Unanswerable cases are split into two flavours, because they fail differently:

* ``absent``      nothing in the corpus is about the topic.
* ``near_miss``   the corpus discusses the topic but never states the specific
                  thing asked (the bonus percentage, a termination penalty,
                  Standard-plan response times for Enterprise...). These are the
                  cases that separate a calibrated system from a cosine cut-off.

The corpus is ``demo_corpus.DEMO_CORPUS`` — short, plain, human-checkable
passages — so a reviewer can verify every label by reading the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cribrix.evaluation.demo_corpus import DEMO_CORPUS
from cribrix.schemas import Chunk

CaseKind = Literal["answerable", "unanswerable", "chitchat"]

CORPUS: list[Chunk] = DEMO_CORPUS


@dataclass(frozen=True)
class GoldenCase:
    """One labelled evaluation example."""

    query: str
    kind: CaseKind
    relevant_ids: frozenset[int] = field(default_factory=frozenset)
    required_facts: tuple[str, ...] = ()
    """Case-insensitive substrings; ANY listed alternative group must match.
    Each entry may contain ``|`` to allow equivalent spellings, e.g. "1000|1,000"."""
    flavour: str = ""
    note: str = ""


def _a(query: str, ids: set[int], *facts: str, note: str = "") -> GoldenCase:
    return GoldenCase(query, "answerable", frozenset(ids), tuple(facts), note=note)


def _u(query: str, flavour: str, note: str = "") -> GoldenCase:
    return GoldenCase(query, "unanswerable", flavour=flavour, note=note)


def _c(query: str) -> GoldenCase:
    return GoldenCase(query, "chitchat")


GOLDEN_CASES: list[GoldenCase] = [
    # ---- answerable -------------------------------------------------------
    _a("What is the refund window for enterprise plans?", {1}, "30 days|30-day|thirty"),
    _a("How long do Standard plan customers have to request a refund?", {2}, "14"),
    _a("Are partial refunds offered on the Pro plan?", {2}, "not|no"),
    _a("How long does it take to process an enterprise refund?", {1}, "10 business days|ten"),
    _a("When are invoices issued?", {3}, "first business day"),
    _a("How is customer data encrypted at rest?", {4}, "AES-256|AES 256"),
    _a("Which TLS version protects data in transit?", {4}, "1.3"),
    _a("How often are encryption keys rotated?", {4}, "90 days|ninety"),
    _a("When did the company complete its SOC 2 Type II audit?", {5}, "March 2026"),
    _a("What is required for production access?", {6}, "multi-factor|MFA"),
    _a("What uptime does the SLA guarantee for Enterprise customers?", {7}, "99.95"),
    _a("What uptime target do Standard plan customers get?", {8}, "99.5"),
    _a("What is the rate limit for the public API?", {9}, "1000|1,000",
       note="A planted forum post claims 10000; triage must drop it."),
    _a("What HTTP status code is returned when the rate limit is exceeded?", {9}, "429"),
    _a("How many times are webhook deliveries retried?", {10}, "five|5"),
    _a("What is the maximum page size for API responses?", {11}, "200"),
    _a("What laptop does the engineering team use?", {12}, "MacBook Pro"),
    _a("What laptops does the marketing team use?", {13}, "MacBook Air"),
    _a("When are salary reviews held?", {16}, "April"),
    _a("How many days of paid annual leave do full-time employees get?", {17}, "25"),
    _a("How many unused leave days can be carried over?", {17}, "5|five"),
    _a("How quickly are new workspaces provisioned?", {18}, "one hour|1 hour"),
    _a("How fast does Enterprise support respond to critical incidents?", {19}, "1 hour|one hour"),
    _a("Which languages is support available in?", {20}, "German", "Japanese"),
    _a("How far in advance must an enterprise contract be cancelled to avoid renewal?",
       {21}, "60 days|sixty"),
    _a("How can a monthly plan be cancelled?", {22}, "account settings"),
    _a("Where is customer data stored?", {23}, "Frankfurt", "Virginia"),
    _a("How often are penetration tests performed?", {24}, "twice"),
    _a("How much notice is given before scheduled maintenance?", {25}, "72 hours"),
    _a("How long does an old API key stay valid after rotation?", {26}, "24 hours"),
    _a("How many days per week can employees work remotely?", {28}, "three|3"),
    _a("How long is parental leave?", {29}, "16 weeks"),
    _a("When is business class travel permitted?", {30}, "eight hours|8 hours"),
    _a("How often are engineering laptops replaced?", {31}, "three years|3 years"),
    # ---- unanswerable: absent from the corpus -----------------------------
    _u("What was the company's total revenue in fiscal year 2019?", "absent"),
    _u("Which airline alliance does the CEO prefer for long-haul travel?", "absent"),
    _u("Who is the current chief financial officer?", "absent"),
    _u("What is the office Wi-Fi password?", "absent"),
    _u("How many employees does the company have?", "absent"),
    _u("What is the company's stock ticker symbol?", "absent"),
    _u("Does the company offer a pension matching scheme?", "absent"),
    _u("What programming language is the backend written in?", "absent"),
    _u("What is the dress code for client meetings?", "absent"),
    _u("Is there free parking at the headquarters?", "absent"),
    # ---- unanswerable: near misses ----------------------------------------
    _u("What is the exact percentage of the annual bonus?", "near_miss",
       "Bonus exists, percentage never stated."),
    _u("What is the penalty for terminating a contract early?", "near_miss",
       "Renewal and refund terms are adjacent; no penalty is stated."),
    _u("What laptop does the design team use?", "near_miss",
       "Engineering and marketing laptops are listed; design is not."),
    _u("What is the refund window for the Business plan?", "near_miss",
       "Enterprise, Standard and Pro are covered; there is no Business plan."),
    _u("How fast does Standard support respond to critical incidents at night?", "near_miss",
       "Standard support is 1 business day; nothing about nights."),
    _u("What uptime percentage does the SLA guarantee for Pro plan customers?", "near_miss",
       "Enterprise and Standard figures exist; Pro is not mentioned."),
    _u("How many sick days do employees get?", "near_miss",
       "Annual and parental leave are covered; sick leave is not."),
    _u("What is the rate limit for the internal admin API?", "near_miss",
       "Only the public API limit is stated."),
    _u("What is the automatic rotation interval for API keys, in days?", "near_miss",
       "Manual rotation is described; no automatic schedule or interval exists."),
    _u("What is the daily meal allowance for business travel?", "near_miss",
       "The expenses policy covers flights and receipts, not meals."),
    _u("When was the last ISO 27001 certification?", "near_miss",
       "Only SOC 2 is mentioned."),
    _u("What does the cafeteria serve on Mondays?", "near_miss",
       "Only Thursdays are mentioned."),
    # ---- chitchat ----------------------------------------------------------
    _c("hey there, good morning!"),
    _c("thanks, that was helpful"),
    _c("Hey, I'm having a rough morning, how are you?"),
    _c("bye, see you soon"),
    _c("hello!"),
    _c("ok great, thank you so much"),
]  # fmt: skip


_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
_SPACES = dict.fromkeys(map(ord, "\u00a0\u202f\u2009\u2007"), " ")


def normalise_text(text: str) -> str:
    """Fold typographic variants LLMs emit (non-breaking hyphens and spaces,
    NFKC compatibility forms) so fact matching compares content, not glyphs."""
    import unicodedata

    return unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_SPACES).lower()


def facts_present(answer: str, facts: tuple[str, ...]) -> bool:
    """True if every required fact (any of its ``|`` alternatives) is in `answer`."""
    text = normalise_text(answer)
    return all(any(normalise_text(alt) in text for alt in fact.split("|")) for fact in facts)
