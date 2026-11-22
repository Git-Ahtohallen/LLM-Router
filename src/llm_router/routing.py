"""Pick a provider for each request from an ordered list of rules.

Rules are checked top to bottom and the first one that matches wins. A rule can match
on the estimated prompt size, on request tags, or both. If no rule matches, the
router's default provider handles the request. The response records which rule fired
in `CompletionResponse.route`, so routing decisions show up in logs and cost reports.

Example::

    router = Router.from_config(
        {"cheap": small_model, "strong": big_model},
        {
            "default": "strong",
            "rules": [
                {"name": "hard-tasks", "target": "strong", "tags": {"complexity": "high"}},
                {"name": "classify", "target": "cheap", "tags": {"task": ["classify", "extract"]}},
                {"name": "short-prompt", "target": "cheap", "max_prompt_tokens": 500},
            ],
        },
    )
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_router.providers.base import Provider
from llm_router.tokens import estimate_request_tokens
from llm_router.types import CompletionRequest, CompletionResponse

DEFAULT_ROUTE = "default"

_RULE_KEYS = frozenset({"name", "target", "min_prompt_tokens", "max_prompt_tokens", "tags"})
_CONFIG_KEYS = frozenset({"default", "rules"})


class RoutingConfigError(ValueError):
    """The routing rules or config are invalid."""


@dataclass(frozen=True, slots=True)
class Rule:
    """Send matching requests to `target`.

    Args:
        name: Recorded in `CompletionResponse.route` when this rule fires.
        target: Key of the provider to use, as passed to `Router`.
        min_prompt_tokens: Match only if the estimated prompt is at least this long.
        max_prompt_tokens: Match only if the estimated prompt is at most this long.
        tags: Every listed tag must be present on the request with one of the
            accepted values. A value can be a single string or a collection of strings.

    A rule with no conditions matches every request.
    """

    name: str
    target: str
    min_prompt_tokens: int | None = None
    max_prompt_tokens: int | None = None
    tags: Mapping[str, str | Collection[str]] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise RoutingConfigError("rule name must not be empty")
        if not self.target:
            raise RoutingConfigError(f"rule {self.name!r}: target must not be empty")
        for label, value in (
            ("min_prompt_tokens", self.min_prompt_tokens),
            ("max_prompt_tokens", self.max_prompt_tokens),
        ):
            if value is not None and value < 0:
                raise RoutingConfigError(f"rule {self.name!r}: {label} must not be negative")
        if (
            self.min_prompt_tokens is not None
            and self.max_prompt_tokens is not None
            and self.min_prompt_tokens > self.max_prompt_tokens
        ):
            raise RoutingConfigError(
                f"rule {self.name!r}: min_prompt_tokens is greater than max_prompt_tokens"
            )
        normalized: dict[str, frozenset[str]] = {}
        for key, accepted in self.tags.items():
            values = frozenset([accepted]) if isinstance(accepted, str) else frozenset(accepted)
            if not values:
                raise RoutingConfigError(f"rule {self.name!r}: tag {key!r} has no accepted values")
            normalized[key] = values
        # Frozen dataclass: normalize in place once so matching stays simple.
        object.__setattr__(self, "tags", normalized)

    def matches(self, request: CompletionRequest, prompt_tokens: int) -> bool:
        if self.min_prompt_tokens is not None and prompt_tokens < self.min_prompt_tokens:
            return False
        if self.max_prompt_tokens is not None and prompt_tokens > self.max_prompt_tokens:
            return False
        return all(request.tags.get(key) in accepted for key, accepted in self.tags.items())


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Why a request went where it went."""

    route: str
    """Name of the rule that fired, or "default"."""
    target: str
    """Key of the chosen provider."""
    provider: Provider
    prompt_tokens: int


class Router:
    """Route each request to one of several providers using ordered rules."""

    def __init__(
        self,
        providers: Mapping[str, Provider],
        rules: Sequence[Rule] = (),
        *,
        default: str,
    ) -> None:
        if not providers:
            raise RoutingConfigError("router needs at least one provider")
        if default not in providers:
            raise RoutingConfigError(f"default provider {default!r} is not configured")
        seen: set[str] = set()
        for rule in rules:
            if rule.name == DEFAULT_ROUTE:
                raise RoutingConfigError(f"rule name {DEFAULT_ROUTE!r} is reserved")
            if rule.name in seen:
                raise RoutingConfigError(f"duplicate rule name {rule.name!r}")
            seen.add(rule.name)
            if rule.target not in providers:
                raise RoutingConfigError(
                    f"rule {rule.name!r} targets unknown provider {rule.target!r}"
                )
        self._providers = dict(providers)
        self._rules = tuple(rules)
        self._default = default

    @classmethod
    def from_config(cls, providers: Mapping[str, Provider], config: Mapping[str, Any]) -> Router:
        """Build a router from plain data (e.g. parsed JSON, YAML or TOML)."""
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            raise RoutingConfigError(f"unknown config keys: {sorted(unknown)}")
        if "default" not in config:
            raise RoutingConfigError("config needs a 'default' provider")
        rules: list[Rule] = []
        for i, raw in enumerate(config.get("rules", [])):
            if not isinstance(raw, Mapping):
                raise RoutingConfigError(f"rule #{i} must be a mapping")
            unknown = set(raw) - _RULE_KEYS
            if unknown:
                raise RoutingConfigError(f"rule #{i}: unknown keys {sorted(unknown)}")
            missing = {"name", "target"} - set(raw)
            if missing:
                raise RoutingConfigError(f"rule #{i}: missing keys {sorted(missing)}")
            rules.append(Rule(**raw))
        return cls(providers, rules, default=config["default"])

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    def route(self, request: CompletionRequest) -> RouteDecision:
        """Decide where a request goes without calling any provider."""
        prompt_tokens = estimate_request_tokens(request)
        for rule in self._rules:
            if rule.matches(request, prompt_tokens):
                return RouteDecision(
                    route=rule.name,
                    target=rule.target,
                    provider=self._providers[rule.target],
                    prompt_tokens=prompt_tokens,
                )
        return RouteDecision(
            route=DEFAULT_ROUTE,
            target=self._default,
            provider=self._providers[self._default],
            prompt_tokens=prompt_tokens,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Route the request, call the chosen provider and record the route taken."""
        decision = self.route(request)
        response = await decision.provider.complete(request)
        return dataclasses.replace(response, route=decision.route)
