"""System-2 generator: the traditional, expensive, free-form LLM.

Supports OpenAI, Anthropic, OpenRouter, Azure/Together/Groq/Ollama and any
other OpenAI-compatible endpoint via a custom base URL.

Why only two transport implementations for five providers
---------------------------------------------------------
OpenAI, OpenRouter, Together, Groq, Fireworks, vLLM, LM Studio and Ollama all
speak the same ``POST /chat/completions`` wire format. Writing a separate SDK
integration per provider would be duplicated code with five times the surface
area to break. ``OpenAICompatibleClient`` covers all of them and differs only
in base URL, auth header and default model.

Anthropic is the genuine exception — different endpoint, different auth header,
a mandatory ``anthropic-version`` header, a top-level ``system`` parameter
rather than a system message, and a different response shape. It gets its own
client.

Raw ``httpx`` is used rather than the vendor SDKs so the container stays small
and the dependency tree doesn't carry three competing HTTP stacks.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Protocol, runtime_checkable

import httpx

from cribrix.clients.recording import RecordingMiss, ResponseCache, request_key
from cribrix.config import Settings
from cribrix.schemas import Chunk


class LLMError(RuntimeError):
    """Raised when the generator fails, times out, or returns an unusable body."""


@runtime_checkable
class LLMClient(Protocol):
    """Structural contract for the System-2 generator."""

    async def generate(
        self, query: str, chunks: list[Chunk], *, system_prompt: str | None = None
    ) -> str:
        """Draft a prose answer to `query` grounded in `chunks`."""
        ...

    async def health(self) -> bool:
        """Cheap liveness probe."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


NAIVE_SYSTEM_PROMPT = """\
You are a helpful assistant. Use the context passages below to answer the \
user's question as helpfully and specifically as possible."""
"""The prompt a typical RAG tutorial ships with.

Used only by the baseline in the scenario suite. It is *not* a strawman: it is
what "helpful assistant + context" looks like in the wild. The omission that
matters is that it never grants permission to refuse, which is precisely the
pressure that makes a model invent a specific figure when asked for one."""


SYSTEM_PROMPT = """\
You are a precise assistant. Answer the user's question using ONLY the numbered \
context passages provided. Do not introduce facts, figures, names or dates that \
do not appear in the context. If the context does not contain the answer, say so \
explicitly rather than estimating or guessing. Keep the answer under three sentences."""


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    """Render the user turn with numbered, attributable context passages."""
    if not chunks:
        return f"Question: {query}\n\nContext: (none available)"
    passages = "\n\n".join(
        f"[{i}] (source={c.document_id}) {c.content}" for i, c in enumerate(chunks, start=1)
    )
    return f"Context passages:\n{passages}\n\nQuestion: {query}\n\nAnswer:"


# Default model per provider, used when no explicit model is configured.
PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    # provider -> (base_url, default_model)
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", "nex-agi/nex-n2.5-pro:free"),
    "anthropic": ("https://api.anthropic.com/v1", "claude-sonnet-4-20250514"),
    "together": ("https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "xai": ("https://api.x.ai/v1", "grok-4.20-0309-non-reasoning"),
    "ollama": ("http://localhost:11434/v1", "llama3.2"),
    "custom": ("", ""),
}


class OpenAICompatibleClient:
    """Chat-completions client for any OpenAI-compatible endpoint.

    Covers OpenAI, OpenRouter, Together, Groq, Ollama, vLLM, LM Studio and
    self-hosted gateways — they differ only in base URL and default model.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 500,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter uses these for attribution; harmless elsewhere.
        headers.setdefault("HTTP-Referer", "https://github.com/david96182/cribrix")
        headers.setdefault("X-Title", "Cribrix")
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s
        )

    async def generate(
        self, query: str, chunks: list[Chunk], *, system_prompt: str | None = None
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(query, chunks)},
            ],
            # Deterministic by default: a RAG pipeline being evaluated for
            # groundedness should not also be a random number generator.
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        content = await self._post_with_retry(payload)

        if not content or not content.strip():
            raise LLMError("LLM returned an empty completion")
        return _strip_reasoning(content)

    # Shared upstream failures (429 rate limit, 502/503 capacity) are extremely
    # common on free/shared tiers and are transient by nature. Retrying them
    # with backoff is the difference between a demo that works and one that
    # fails randomly in front of an audience.
    _RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    async def _post_with_retry(self, payload: dict[str, Any], attempts: int = 4) -> str:
        """POST a completion, retrying transient upstream failures."""
        last = "unknown error"
        for attempt in range(attempts):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last = f"transport error: {exc}"
            else:
                if response.status_code == 200:
                    body = response.json()
                    # Some gateways return HTTP 200 with an error envelope.
                    if isinstance(body, dict) and body.get("error"):
                        last = f"upstream error: {str(body['error'])[:200]}"
                    else:
                        try:
                            return str(body["choices"][0]["message"]["content"])
                        except (KeyError, IndexError, TypeError) as exc:
                            raise LLMError(
                                f"unexpected LLM response shape: {response.text[:300]}"
                            ) from exc
                elif response.status_code in self._RETRYABLE_STATUS:
                    last = f"HTTP {response.status_code}: {response.text[:200]}"
                else:
                    raise LLMError(
                        f"LLM returned HTTP {response.status_code}: {response.text[:300]}"
                    )
            if attempt < attempts - 1:
                await asyncio.sleep(1.5 * (2**attempt))  # 1.5s, 3s, 6s
        raise LLMError(f"LLM failed after {attempts} attempts: {last}")

    async def health(self) -> bool:
        try:
            response = await self._client.get("/models")
        except httpx.HTTPError:
            return False
        return response.status_code < 500

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicClient:
    """Client for Anthropic's Messages API.

    Kept separate because Anthropic genuinely differs: `x-api-key` instead of
    bearer auth, a required `anthropic-version` header, `system` as a
    top-level field rather than a message, and a `content[].text` response.
    """

    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.anthropic.com/v1",
        model: str = "claude-sonnet-4-20250514",
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        self._model = model
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "x-" + "api-key": api_key,
                "anthropic-version": self.API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=timeout_s,
        )

    async def generate(
        self, query: str, chunks: list[Chunk], *, system_prompt: str | None = None
    ) -> str:
        payload = {
            "model": self._model,
            "system": system_prompt or SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_prompt(query, chunks)}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        try:
            response = await self._client.post("/messages", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(f"Anthropic returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            blocks = response.json()["content"]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"unexpected Anthropic response: {response.text[:300]}") from exc

        if not text.strip():
            raise LLMError("Anthropic returned an empty completion")
        return text.strip()

    async def health(self) -> bool:
        try:
            response = await self._client.get("/models")
        except httpx.HTTPError:
            return False
        return response.status_code < 500

    async def aclose(self) -> None:
        await self._client.aclose()


_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought blocks some open models emit.

    Several free/open models wrap internal reasoning in `<think>` tags. Left in
    place, that text would be fact-checked as if it were part of the answer —
    producing spurious groundedness failures on speculative reasoning the model
    never intended to assert.
    """
    return _THINK_BLOCK.sub("", text).strip()


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class FakeLLMClient:
    """Deterministic extractive generator for offline tests.

    Composes answers **only** from the supplied context, so it is grounded by
    construction. That property is what makes verification tests meaningful: a
    mock that invented text would fail the gate every time and prove nothing
    about the gate's logic. `hallucinate=True` flips it into fabricating a
    number, which is how the gate's positive case is tested.
    """

    model = "fake-extractive"

    def __init__(
        self, *, latency_s: float = 0.0, fail: bool = False, hallucinate: bool = False
    ) -> None:
        self._latency_s = latency_s
        self._fail = fail
        self._hallucinate = hallucinate
        self.call_count = 0
        self.last_chunks: list[Chunk] = []

    async def generate(
        self, query: str, chunks: list[Chunk], *, system_prompt: str | None = None
    ) -> str:
        self.call_count += 1
        self.last_chunks = list(chunks)
        if self._latency_s:
            await asyncio.sleep(self._latency_s)
        if self._fail:
            raise LLMError("FakeLLMClient configured to fail")
        if not chunks:
            return "I don't have enough information to answer that."

        terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 3}
        scored: list[tuple[int, str]] = []
        for chunk in chunks:
            for sentence in _SENTENCE_RE.split(chunk.content.strip()):
                clean = sentence.strip()
                if clean:
                    overlap = len(terms & set(re.findall(r"[a-z0-9]+", clean.lower())))
                    scored.append((overlap, clean))
        scored.sort(key=lambda p: p[0], reverse=True)
        picked = [s for _, s in scored[:2]] or [chunks[0].content.strip()]
        answer = " ".join(s if s.endswith((".", "!", "?")) else f"{s}." for s in picked)

        if self._hallucinate:
            answer += " The exact figure is 10 percent."
        return answer

    async def health(self) -> bool:
        return not self._fail

    async def aclose(self) -> None:
        return None


class RecordingLLMClient:
    """Record live generations to disk, or replay them without a network.

    The key covers the model, system prompt, question and the exact evidence
    passed in, so a change in what triage keeps produces a miss rather than a
    stale draft. Generation failures are recorded too: replaying a run must
    reproduce its GENERATION_FAILED outcomes, not quietly retry them.
    """

    def __init__(
        self, cache: ResponseCache, inner: LLMClient | None = None, *, retry_errors: bool = False
    ) -> None:
        if cache.mode == "record" and inner is None:
            raise ValueError("record mode needs a live client to record from")
        self._cache = cache
        self._inner = inner
        self._retry_errors = retry_errors
        self.model = getattr(inner, "model", None) or cache.meta.get("llm_model", "unknown")
        if cache.mode == "record":
            cache.update_meta(llm_model=self.model)

    async def generate(
        self, query: str, chunks: list[Chunk], *, system_prompt: str | None = None
    ) -> str:
        key = request_key(
            {
                "kind": "llm",
                "model": self.model,
                "system": system_prompt or SYSTEM_PROMPT,
                "query": query,
                "chunks": [c.content for c in chunks],
            }
        )
        hit = self._cache.get(key)
        if hit is not None and "error" in hit and self._retry_errors and self._inner is not None:
            hit = None  # re-attempt transient upstream failures when re-recording
        if hit is None:
            if self._cache.mode == "replay":
                raise RecordingMiss(
                    "LLM generation was not recorded. Re-record with `make eval-record`."
                )
            assert self._inner is not None
            try:
                text = await self._inner.generate(query, chunks, system_prompt=system_prompt)
            except LLMError as exc:
                self._cache.put(key, {"error": str(exc)[:300]})
                raise
            self._cache.put(key, {"text": text})
            return text
        if "error" in hit:
            raise LLMError(f"(replayed) {hit['error']}")
        return str(hit["text"])

    async def health(self) -> bool:
        return True if self._inner is None else await self._inner.health()

    async def aclose(self) -> None:
        self._cache.flush()
        if self._inner is not None:
            await self._inner.aclose()


def build_llm_client(
    settings: Settings, *, cache: ResponseCache | None = None, retry_errors: bool = False
) -> LLMClient:
    """Construct the generator, optionally wrapped for record/replay."""
    if cache is not None and cache.mode == "replay":
        return RecordingLLMClient(cache)
    client = _build_llm_client(settings)
    if cache is None:
        return client
    return RecordingLLMClient(cache, client, retry_errors=retry_errors)


def _build_llm_client(settings: Settings) -> LLMClient:
    """Construct the generator described by configuration.

    Raises:
        LLMError: if a live provider is selected without the required
            credentials or, for `custom`, without a base URL.
    """
    provider = settings.llm_provider
    if provider == "fake":
        return FakeLLMClient()

    if not settings.llm_api_key:
        raise LLMError(
            f"CRIBRIX_LLM_API_KEY is required for provider {provider!r}. "
            "Set CRIBRIX_LLM_PROVIDER=fake for offline use."
        )

    default_url, default_model = PROVIDER_DEFAULTS.get(provider, ("", ""))
    base_url = settings.llm_base_url or default_url
    model = settings.llm_model or default_model

    if not base_url:
        raise LLMError("CRIBRIX_LLM_BASE_URL is required when CRIBRIX_LLM_PROVIDER=custom")
    if not model:
        raise LLMError(f"CRIBRIX_LLM_MODEL is required for provider {provider!r}")

    if provider == "anthropic":
        return AnthropicClient(
            settings.llm_api_key,
            base_url=base_url,
            model=model,
            timeout_s=settings.llm_timeout_s,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    return OpenAICompatibleClient(
        settings.llm_api_key,
        base_url=base_url,
        model=model,
        timeout_s=settings.llm_timeout_s,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
