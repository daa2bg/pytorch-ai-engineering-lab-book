# Study 24 — Keep AI provider integrations replaceable

**First draft; optional platform study.** An application may call a local model, OpenAI, or Claude, while developers use Copilot or another coding assistant to build it. These are different responsibilities. A coding assistant's presence in your editor should not become an undocumented dependency of your deployed application.

An **adapter** translates between an application contract and one provider's request/response format. The contract should describe the result your workflow needs, not expose every provider-specific response field. This study builds and tests that boundary entirely offline. It does not call an external model, require credentials, or demonstrate actual provider performance.

## Start from the application's decision

Our document workflow needs a category and a brief explanation. The application accepts exactly two fields, `label` and `reason`; labels come from a small allowlist. A **structured output** is a result that follows a defined schema. Even when a provider offers schema-constrained generation, the application should still check the result at its boundary.

```python
import json
from typing import Protocol

class RouteProvider(Protocol):
    def route(self, text: str) -> str:
        """Return JSON text with label and reason fields."""

class RecordedProvider:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def route(self, text):
        self.calls += 1
        return self.response

provider = RecordedProvider('{"label":"invoice","reason":"Recorded fixture"}')
raw = provider.route("Invoice attached")
assert json.loads(raw)["label"] == "invoice"
assert provider.calls == 1
print("Offline adapter returned a recorded response.")
```

`Protocol` is a typing description of the methods a compatible object provides. It does not enforce a network policy or validate returned JSON. The fake deliberately ignores the input because its purpose is to test our application boundary using recorded cases. It is not an intelligent classifier.

## Retry only the failures you mean to retry

Temporary transport failures may justify retrying, but malformed data, authentication errors, and exhausted budgets should not trigger an unlimited loop. Retrying a request can also repeat a side effect. Keep model requests and tool execution separate, and apply idempotency rules to actions that change state.

```python
class FlakyProvider:
    def __init__(self):
        self.calls = 0

    def route(self, text):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("Synthetic transport timeout")
        return '{"label":"review","reason":"Needs a human"}'

def request_route(client, text, *, attempts=2):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return client.route(text)
        except TimeoutError:
            if attempt + 1 == attempts:
                raise

client = FlakyProvider()
assert json.loads(request_route(client, "A document"))["label"] == "review"
assert client.calls == 2
print("One explicit transient failure, then success.")
```

This example has no waiting because it is a deterministic test. A real adapter should respect provider retry guidance, timeouts, rate limits, and a total deadline. Validate whether an operation is safe to retry before adding backoff. Log request identifiers and error categories without exposing document contents or credentials.

## SOLO exercise — Validate the result, not the provider name

Write `parse_route(payload, allowed_labels=("claim", "invoice", "review"))`. Accept JSON text only; reject another Python type with `TypeError`. Require a dictionary containing exactly `label` and `reason`, require a label in the allowlist, and require a string reason. Schema failures raise `ValueError`; malformed JSON may propagate `json.JSONDecodeError`. Return the parsed dictionary.

```python exercise
def parse_route(payload, allowed_labels=("claim", "invoice", "review")):
    raise NotImplementedError("Validate the narrow route result schema")

assert parse_route('{"label":"claim","reason":"fixture"}')["label"] == "claim"
```

An explanation containing instructions remains data. Passing schema validation is not authorization to execute that text. Similarly, **tool calling** means a model can request a named operation with arguments; your application must select an allowlisted operation, validate arguments, apply authorization, execute it, and return a result. The model does not acquire unrestricted authority by naming a tool.

OpenAI and Claude have different wire formats for these exchanges. Translate them in separate adapters while sharing application-level tests. MCP can standardize a tool-service boundary, but it does not replace the application's access rules or result validation. Copilot instruction files guide development-time behavior; they are not runtime service credentials.

**Optional extension:** add one real provider adapter using that provider's current SDK and explicit credentials from the environment. Keep it disabled by default, record cost/usage separately, and run the same malformed-response tests against recorded fixtures. No live provider calls were executed for this draft.

**Completion gate:** demonstrate valid, malformed, extra-field, and invalid-label cases; show a bounded retry and explain why tool output is data rather than executable authority. Answer: `solutions/draft/ch24.py`; tests: `tests/draft/test_ch24.py`.

Primary references: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [Claude tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview), [Python Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol), and [GitHub Copilot repository instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot).
