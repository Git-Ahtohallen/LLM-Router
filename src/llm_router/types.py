"""Core request/response types shared by providers, the router and the cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """A provider-agnostic chat completion request."""

    messages: tuple[Message, ...]
    max_tokens: int = 512
    temperature: float = 0.0
    # Free-form labels (e.g. task="summarize") used by routing rules and reporting.
    tags: dict[str, str] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("CompletionRequest needs at least one message")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        tags: dict[str, str] | None = None,
    ) -> CompletionRequest:
        """Convenience constructor for the common single-prompt case."""
        msgs: list[Message] = []
        if system:
            msgs.append(Message("system", system))
        msgs.append(Message("user", prompt))
        return cls(
            messages=tuple(msgs),
            max_tokens=max_tokens,
            temperature=temperature,
            tags=dict(tags or {}),
        )

    @property
    def prompt_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    """What every provider returns, normalized."""

    text: str
    provider: str
    model: str
    usage: Usage
    latency_ms: float
    cost_usd: float = 0.0
    cached: bool = False
