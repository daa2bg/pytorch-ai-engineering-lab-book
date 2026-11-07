# Study 16 — Put a Stable Boundary Around a Model

**First draft. Core study, about 40 minutes.** A notebook function becomes useful to another application when it has an explicit request and response contract. The goal here is a small prediction boundary that you can test without starting a server. HTTP is an adapter around that boundary, not the place to hide preprocessing or model logic.

## Separate application behavior from transport

An inference service needs to know which fields are allowed, what a valid text looks like, and which labels can be returned. A function taking a dictionary is enough to practice those choices. The predictor below is a transparent keyword rule so that service behavior is testable without model files. Replace it with your loaded model later; do not mistake its outputs for an evaluated ML result.

```python
def keyword_predictor(text):
    words = text.lower().split()
    if "invoice" in words:
        return "invoice"
    if "claim" in words:
        return "claim"
    return "general"

assert keyword_predictor("Invoice due") == "invoice"
assert keyword_predictor("Please review my claim") == "claim"
```

An application function should reject a bad request before invoking the expensive predictor. Here a valid request has exactly one key, `text`; its value is a string, contains a non-whitespace character, and has at most 200 characters. A size bound is a resource contract, not a comprehensive abuse defense. A tokenizer-based production service may also limit tokens and total batch size.

```python
def handle_prediction(payload):
    if not isinstance(payload, dict):
        raise TypeError("request must be a dictionary")
    if set(payload) != {"text"}:
        raise ValueError("request must contain exactly text")
    text = payload["text"]
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip() or len(text) > 200:
        raise ValueError("invalid text length")
    return {"label": keyword_predictor(text)}

request = {"text": "Invoice due"}
assert handle_prediction(request) == {"label": "invoice"}
assert request == {"text": "Invoice due"}
print(handle_prediction(request))
```

`set(payload)` visits dictionary keys, so comparison with `{"text"}` rejects both missing and extra fields. Explicitly validating the input means an accidental `authorization` or `customer_id` field does not silently enter model processing. It does not mean the application should echo rejected values into logs or errors.

## Add FastAPI as an adapter

FastAPI uses Python type declarations and Pydantic models to interpret request bodies. This complete optional example constructs an app but starts no listener. Save the app in a module if you later choose to run an ASGI server. Core validation above remains runnable without FastAPI.

```python
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, ConfigDict
except ModuleNotFoundError:
    print("SKIP HTTP adapter: optional FastAPI/Pydantic dependencies are absent.")
else:
    class PredictionRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        text: str = Field(min_length=1, max_length=200)

    app = FastAPI()

    @app.post("/predict")
    def predict_endpoint(body: PredictionRequest):
        try:
            return handle_prediction({"text": body.text})
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="Invalid prediction request") from error

    assert predict_endpoint(PredictionRequest(text="Invoice due")) == {"label": "invoice"}
    print("App constructed without starting a server.")
```

This is an application adapter, not a production deployment. A deployed service also needs trusted model loading, model/version readiness, authentication, resource limits, timeouts, error handling, and operational monitoring. Load a trusted model once per worker rather than rebuilding it for each request. Avoid doing CPU-heavy inference directly in an asynchronous event loop just because `async def` is available.

### SOLO exercise — inject the actual predictor

Implement `predict_request(payload, predictor, *, max_chars=200)`. Follow the dictionary/text contract above, but call the supplied predictor. The caller promises a positive integer `max_chars`. Verify that its result is one of `claim`, `invoice`, or `general`; otherwise raise `ValueError`. Return exactly `{"label": result}` without editing the request. Reject invalid request types with `TypeError` and invalid fields/lengths with `ValueError`.

```python exercise
def predict_request(payload, predictor, *, max_chars=200):
    raise NotImplementedError("Implement a tested prediction boundary")

assert predict_request({"text": "hello"}, lambda text: "general") == {"label": "general"}
```

**Completion gate:** prove malformed input does not call the predictor, valid input is preserved, and an unknown model label fails visibly. **Optional study:** use FastAPI's `TestClient` with optional HTTPX installed to assert HTTP status codes and response bodies in-process. A direct endpoint-function call does not test JSON parsing or middleware, so record that distinction in your validation evidence.

Reference answers: `solutions/draft/ch16.py`; tests: `tests/draft/test_ch16.py`. Primary sources: [FastAPI request bodies](https://fastapi.tiangolo.com/tutorial/body/) and [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/). Next, persist the workflow state surrounding a prediction so a process can restart without forgetting completed work.
