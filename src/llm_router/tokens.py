"""Cheap, dependency-free token estimates.

Real tokenizers differ per model, but routing only needs a rough size signal, and the
mock provider needs a stable count. Four characters per token is a common rule of
thumb for English text.
"""

from __future__ import annotations

from llm_router.types import CompletionRequest

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_request_tokens(request: CompletionRequest) -> int:
    """Estimated input tokens for a whole request (all messages)."""
    return sum(estimate_tokens(m.content) for m in request.messages)
