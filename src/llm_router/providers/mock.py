"""An offline provider for tests, local development and CI.

It never calls the network. Responses are deterministic for a given request, and
you can inject latency and failures to exercise routing, caching and fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable

from llm_router.providers.base import ModelPricing, ProviderError
from llm_router.types import CompletionRequest, CompletionResponse, Usage

# Rough rule of thumb for English text; good enough for a mock and for routing.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class MockProvider:
    """A fake model endpoint.

    Args:
        name: Provider name reported in responses.
        model: Model name reported in responses.
        pricing: Prices used to compute `cost_usd`.
        latency_ms: Simulated latency per call.
        fail_times: Fail this many calls before succeeding (simulates an outage).
        fail_retryable: Whether injected failures are retryable.
        responder: Optional function that builds the reply text from the request.
    """

    def __init__(
        self,
        name: str = "mock",
        model: str = "mock-small",
        *,
        pricing: ModelPricing | None = None,
        latency_ms: float = 0.0,
        fail_times: int = 0,
        fail_retryable: bool = True,
        responder: Callable[[CompletionRequest], str] | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._pricing = pricing or ModelPricing(input_per_mtok=0.0, output_per_mtok=0.0)
        self._latency_ms = latency_ms
        self._fail_remaining = fail_times
        self._fail_retryable = fail_retryable
        self._responder = responder or self._default_reply
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def pricing(self) -> ModelPricing:
        return self._pricing

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        started = time.perf_counter()

        if self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000)

        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise ProviderError(self._name, "simulated outage", retryable=self._fail_retryable)

        text = self._responder(request)
        # Respect max_tokens the way a real model would: truncate the output.
        max_chars = request.max_tokens * CHARS_PER_TOKEN
        text = text[:max_chars]

        usage = Usage(
            input_tokens=sum(estimate_tokens(m.content) for m in request.messages),
            output_tokens=estimate_tokens(text),
        )
        return CompletionResponse(
            text=text,
            provider=self._name,
            model=self._model,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=self._pricing.cost(usage.input_tokens, usage.output_tokens),
        )

    def _default_reply(self, request: CompletionRequest) -> str:
        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        digest = hashlib.sha256(last_user.encode()).hexdigest()[:8]
        return f"[{self._model}:{digest}] {last_user[:200]}"
