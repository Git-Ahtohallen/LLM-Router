import pytest

from llm_router import Router, RoutingConfigError, Rule
from llm_router.providers import MockProvider
from llm_router.types import CompletionRequest


@pytest.fixture
def providers() -> dict[str, MockProvider]:
    return {
        "cheap": MockProvider(name="mock", model="mock-small"),
        "strong": MockProvider(name="mock", model="mock-large"),
    }


@pytest.fixture
def router(providers: dict[str, MockProvider]) -> Router:
    return Router(
        providers,
        [
            Rule("hard-tasks", "strong", tags={"complexity": "high"}),
            Rule("classify", "cheap", tags={"task": ["classify", "extract"]}),
            Rule("short-prompt", "cheap", max_prompt_tokens=100),
        ],
        default="strong",
    )


def test_short_prompt_goes_to_cheap(router: Router) -> None:
    decision = router.route(CompletionRequest.from_prompt("x" * 400))  # 100 tokens
    assert (decision.route, decision.target) == ("short-prompt", "cheap")
    assert decision.prompt_tokens == 100


def test_long_prompt_falls_through_to_default(router: Router) -> None:
    decision = router.route(CompletionRequest.from_prompt("x" * 404))  # 101 tokens
    assert (decision.route, decision.target) == ("default", "strong")


def test_tag_rule_beats_size_rule_by_order(router: Router) -> None:
    req = CompletionRequest.from_prompt("short", tags={"complexity": "high"})
    assert router.route(req).route == "hard-tasks"


def test_tag_rule_accepts_any_listed_value(router: Router) -> None:
    long_text = "x" * 4000
    for task in ("classify", "extract"):
        req = CompletionRequest.from_prompt(long_text, tags={"task": task})
        assert router.route(req).route == "classify"
    other = CompletionRequest.from_prompt(long_text, tags={"task": "summarize"})
    assert router.route(other).route == "default"


def test_prompt_size_counts_all_messages(router: Router) -> None:
    req = CompletionRequest.from_prompt("x" * 200, system="y" * 208)  # 50 + 52 tokens
    assert router.route(req).route == "default"


def test_rule_needs_every_condition() -> None:
    rule = Rule("r", "cheap", min_prompt_tokens=10, max_prompt_tokens=20, tags={"task": "chat"})
    chat = CompletionRequest.from_prompt("x", tags={"task": "chat"})
    assert rule.matches(chat, 15)
    assert not rule.matches(chat, 9)
    assert not rule.matches(chat, 21)
    assert not rule.matches(CompletionRequest.from_prompt("x"), 15)


async def test_complete_calls_chosen_provider_and_records_route(
    router: Router, providers: dict[str, MockProvider]
) -> None:
    resp = await router.complete(CompletionRequest.from_prompt("hi"))
    assert resp.route == "short-prompt"
    assert resp.model == "mock-small"
    assert providers["cheap"].calls == 1
    assert providers["strong"].calls == 0


async def test_direct_provider_calls_have_no_route(providers: dict[str, MockProvider]) -> None:
    resp = await providers["cheap"].complete(CompletionRequest.from_prompt("hi"))
    assert resp.route is None


def test_from_config(providers: dict[str, MockProvider]) -> None:
    router = Router.from_config(
        providers,
        {
            "default": "strong",
            "rules": [{"name": "short", "target": "cheap", "max_prompt_tokens": 50}],
        },
    )
    assert [r.name for r in router.rules] == ["short"]
    assert router.route(CompletionRequest.from_prompt("hi")).target == "cheap"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"rules": []}, "'default'"),
        ({"default": "strong", "extra": 1}, "unknown config keys"),
        ({"default": "strong", "rules": ["nope"]}, "must be a mapping"),
        ({"default": "strong", "rules": [{"name": "a"}]}, "missing keys"),
        ({"default": "strong", "rules": [{"name": "a", "target": "cheap", "x": 1}]}, "unknown"),
    ],
)
def test_from_config_rejects_bad_config(
    providers: dict[str, MockProvider], config: dict[str, object], message: str
) -> None:
    with pytest.raises(RoutingConfigError, match=message):
        Router.from_config(providers, config)


@pytest.mark.parametrize(
    ("rules", "default", "message"),
    [
        ([], "missing", "default provider"),
        ([Rule("a", "missing")], "strong", "unknown provider"),
        ([Rule("a", "cheap"), Rule("a", "strong")], "strong", "duplicate"),
        ([Rule("default", "cheap")], "strong", "reserved"),
    ],
)
def test_router_validates_rules(
    providers: dict[str, MockProvider], rules: list[Rule], default: str, message: str
) -> None:
    with pytest.raises(RoutingConfigError, match=message):
        Router(providers, rules, default=default)


def test_router_needs_providers() -> None:
    with pytest.raises(RoutingConfigError, match="at least one provider"):
        Router({}, default="x")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": ""}, "name"),
        ({"target": ""}, "target"),
        ({"min_prompt_tokens": -1}, "negative"),
        ({"min_prompt_tokens": 10, "max_prompt_tokens": 5}, "greater than"),
        ({"tags": {"task": []}}, "no accepted values"),
    ],
)
def test_invalid_rules_are_rejected(kwargs: dict[str, object], message: str) -> None:
    args: dict[str, object] = {"name": "r", "target": "cheap", **kwargs}
    with pytest.raises(RoutingConfigError, match=message):
        Rule(**args)  # type: ignore[arg-type]
