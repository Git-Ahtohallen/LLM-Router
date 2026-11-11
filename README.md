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

## Roadmap

- [x] Core types and provider interface, mock provider
- [ ] Routing by task size
- [ ] Response cache
- [ ] Provider fallback
- [ ] Cost tracking and reporting
- [ ] Real providers (OpenAI, Anthropic)
- [ ] Eval suite
- [ ] CI

## License

[MIT](LICENSE)
