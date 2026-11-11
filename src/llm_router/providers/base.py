"""The interface every model provider implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from llm_router.types import CompletionRequest, CompletionResponse


class ProviderError(Exception):
    """A provider call failed.

    `retryable` tells the router whether trying another provider (or the same one
    again) makes sense: outages, rate limits and timeouts are retryable; a bad
    request is not.
    """

    def __init__(self, provider: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD price per one million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


@runtime_checkable
class Provider(Protocol):
    """A single model endpoint (one provider + one model)."""

    @property
    def name(self) -> str:
        """Provider name, e.g. "openai" or "mock"."""
        ...

    @property
    def model(self) -> str:
        """Model identifier sent to the provider."""
        ...

    @property
    def pricing(self) -> ModelPricing: ...

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run one completion. Raises ProviderError on failure."""
        ...
