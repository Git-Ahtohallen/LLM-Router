import pytest

from llm_router.providers import ModelPricing
from llm_router.types import CompletionRequest, Message, Usage


def test_from_prompt_builds_system_and_user_messages() -> None:
    req = CompletionRequest.from_prompt("hi", system="be brief", tags={"task": "chat"})
    assert req.messages == (Message("system", "be brief"), Message("user", "hi"))
    assert req.tags == {"task": "chat"}


def test_requests_with_same_content_are_equal_regardless_of_tags() -> None:
    # Tags are metadata; they must not change request identity (important for caching).
    a = CompletionRequest.from_prompt("same", tags={"task": "a"})
    b = CompletionRequest.from_prompt("same", tags={"task": "b"})
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"messages": ()}, "at least one message"),
        ({"messages": (Message("user", "x"),), "max_tokens": 0}, "max_tokens"),
        ({"messages": (Message("user", "x"),), "temperature": 3.0}, "temperature"),
    ],
)
def test_invalid_requests_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CompletionRequest(**kwargs)  # type: ignore[arg-type]


def test_usage_total() -> None:
    assert Usage(input_tokens=10, output_tokens=5).total_tokens == 15


def test_pricing_is_per_million_tokens() -> None:
    pricing = ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)
    assert pricing.cost(1_000_000, 0) == pytest.approx(3.0)
    assert pricing.cost(1_000, 1_000) == pytest.approx(0.018)
