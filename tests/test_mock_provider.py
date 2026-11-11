import pytest

from llm_router.providers import MockProvider, ModelPricing, Provider, ProviderError
from llm_router.types import CompletionRequest


async def test_mock_satisfies_provider_protocol() -> None:
    assert isinstance(MockProvider(), Provider)


async def test_reply_is_deterministic_for_same_prompt() -> None:
    provider = MockProvider(model="mock-small")
    req = CompletionRequest.from_prompt("Summarize this visit note.")
    first = await provider.complete(req)
    second = await provider.complete(req)
    assert first.text == second.text
    assert first.model == "mock-small"
    assert first.provider == "mock"
    assert provider.calls == 2


async def test_different_prompts_get_different_replies() -> None:
    provider = MockProvider()
    a = await provider.complete(CompletionRequest.from_prompt("one"))
    b = await provider.complete(CompletionRequest.from_prompt("two"))
    assert a.text != b.text


async def test_usage_and_cost_are_reported() -> None:
    provider = MockProvider(pricing=ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0))
    resp = await provider.complete(CompletionRequest.from_prompt("x" * 400))
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens > 0
    expected = (resp.usage.input_tokens * 1.0 + resp.usage.output_tokens * 2.0) / 1_000_000
    assert resp.cost_usd == pytest.approx(expected)
    assert resp.cached is False


async def test_output_respects_max_tokens() -> None:
    provider = MockProvider(responder=lambda _req: "word " * 1000)
    resp = await provider.complete(CompletionRequest.from_prompt("long", max_tokens=10))
    assert resp.usage.output_tokens <= 10


async def test_injected_failures_then_recovery() -> None:
    provider = MockProvider(fail_times=2)
    req = CompletionRequest.from_prompt("hello")
    for _ in range(2):
        with pytest.raises(ProviderError) as exc:
            await provider.complete(req)
        assert exc.value.retryable is True
        assert exc.value.provider == "mock"
    resp = await provider.complete(req)
    assert resp.text
    assert provider.calls == 3


async def test_non_retryable_failure_flag() -> None:
    provider = MockProvider(fail_times=1, fail_retryable=False)
    with pytest.raises(ProviderError) as exc:
        await provider.complete(CompletionRequest.from_prompt("bad"))
    assert exc.value.retryable is False


async def test_simulated_latency_is_measured() -> None:
    provider = MockProvider(latency_ms=30)
    resp = await provider.complete(CompletionRequest.from_prompt("slow"))
    assert resp.latency_ms >= 25
