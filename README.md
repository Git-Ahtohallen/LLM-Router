# LLM-Router

A small, production-minded router for LLM requests:

- **Route by task size**: send short, simple tasks to a cheaper model and long or complex ones to a stronger model.
- **Cache repeat prompts**: skip the model call entirely when the same request was answered recently.
- **Fail over across providers**: if one provider errors or times out, retry on another.
- **Track cost per request**: every response carries token usage and a dollar cost.
- **Evaluate changes**: a pytest eval set guards against regressions when prompts or routing rules change.

> Status: early development. See the roadmap below.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Everything runs offline against a built-in mock provider, so no API keys are needed.

## Routing

A `Router` holds named providers and an ordered list of rules. The first rule that matches
picks the provider; if none match, the default provider handles the request. Rules can match
on estimated prompt size (`min_prompt_tokens` / `max_prompt_tokens`), on request tags, or both.

```python
from llm_router import Router
from llm_router.providers import MockProvider
from llm_router.types import CompletionRequest

router = Router.from_config(
    {"cheap": MockProvider(model="mock-small"), "strong": MockProvider(model="mock-large")},
    {
        "default": "strong",
        "rules": [
            {"name": "hard-tasks", "target": "strong", "tags": {"complexity": "high"}},
            {"name": "classify", "target": "cheap", "tags": {"task": ["classify", "extract"]}},
            {"name": "short-prompt", "target": "cheap", "max_prompt_tokens": 500},
        ],
    },
)

resp = await router.complete(CompletionRequest.from_prompt("Is this note about diabetes?"))
resp.model  # "mock-small"
resp.route  # "short-prompt"
```

The config is plain data, so it can live in JSON, YAML or TOML. Every response records the
rule that fired in `resp.route`, and `router.route(request)` returns the decision without
calling a provider.

## Roadmap

- [x] Core types and provider interface, mock provider
- [x] Routing by task size
- [ ] Response cache
- [ ] Provider fallback
- [ ] Cost tracking and reporting
- [ ] Real providers (OpenAI, Anthropic)
- [ ] Eval suite
- [ ] CI

## License

[MIT](LICENSE)
